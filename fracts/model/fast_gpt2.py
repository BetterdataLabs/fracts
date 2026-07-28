"""
CUDA-graph capturable GPT-2 decode step over a preallocated attention window.

Sampling one token through `GPT2Model` costs ~4 ms wall-clock while the GPU work is
well under 0.5 ms: a 12-layer block issues roughly 240 kernels, and HuggingFace adds
per-layer Python on top. Replaying the step as a CUDA graph removes that, but capture
needs shapes that never change, which HuggingFace's growing KV cache cannot give.

So the window is preallocated at its full width and attention runs over all of it with
a mask, while the query stays a single token. That keeps the arithmetic identical to a
cached step -- one token of projections and MLP -- and makes every shape static.

Only the incremental path is covered; training and full-sequence scoring keep using
`GPT2Model` directly.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StaticWindowDecoder:
    """Decode one token at a time against a fixed-size KV window."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        capacity: int,
        max_len: int,
    ):
        config = transformer.config
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        self.eps = config.layer_norm_epsilon
        self.capacity = capacity
        self.max_len = max_len

        self.wpe = transformer.wpe.weight
        self.ln_f = (transformer.ln_f.weight, transformer.ln_f.bias)
        self.layers: List[Tuple[torch.Tensor, ...]] = [
            (
                block.ln_1.weight,
                block.ln_1.bias,
                block.attn.c_attn.weight,
                block.attn.c_attn.bias,
                block.attn.c_proj.weight,
                block.attn.c_proj.bias,
                block.ln_2.weight,
                block.ln_2.bias,
                block.mlp.c_fc.weight,
                block.mlp.c_fc.bias,
                block.mlp.c_proj.weight,
                block.mlp.c_proj.bias,
            )
            for block in transformer.h
        ]

        device = self.wpe.device
        dtype = self.wpe.dtype
        shape = (len(self.layers), capacity, self.num_heads, max_len, self.head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.write_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.window = torch.arange(max_len, device=device)

        self.length = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_in = torch.zeros(capacity, 1, self.embed_dim, device=device, dtype=dtype)
        self.static_out: Optional[torch.Tensor] = None

    def reset(self) -> None:
        """Start a new sequence. Stale window entries stay masked out."""
        self.length = 0

    def _block(self, idx: int, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        (
            ln1_w, ln1_b, qkv_w, qkv_b, o_w, o_b,
            ln2_w, ln2_b, fc_w, fc_b, proj_w, proj_b,
        ) = self.layers[idx]
        b, e = self.capacity, self.embed_dim

        residual = h
        x = F.layer_norm(h, (e,), ln1_w, ln1_b, self.eps).view(b, e)
        qkv = torch.addmm(qkv_b, x, qkv_w)  # B, 3E
        q, k, v = qkv.split(e, dim=-1)
        q = q.view(b, self.num_heads, 1, self.head_dim)
        k = k.view(b, self.num_heads, 1, self.head_dim)
        v = v.view(b, self.num_heads, 1, self.head_dim)

        k_all, v_all = self.k_cache[idx], self.v_cache[idx]
        k_all.index_copy_(2, self.write_pos, k)
        v_all.index_copy_(2, self.write_pos, v)

        attn = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(b, e)
        h = torch.addmm(o_b, attn, o_w).view(b, 1, e) + residual

        residual = h
        x = F.layer_norm(h, (e,), ln2_w, ln2_b, self.eps).view(b, e)
        x = torch.addmm(fc_b, x, fc_w)
        x = F.gelu(x)
        x = torch.addmm(proj_b, x, proj_w).view(b, 1, e)
        return residual + x

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.wpe.index_select(0, self.write_pos).unsqueeze(0)
        mask = (self.window <= self.write_pos).view(1, 1, 1, self.max_len)
        for idx in range(len(self.layers)):
            h = self._block(idx, h, mask)
        return F.layer_norm(h, (self.embed_dim,), *self.ln_f, self.eps)

    def _capture(self) -> None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._decode(self.static_in)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.static_out = self._decode(self.static_in)
        self.graph = graph
        logger.info(
            f"Captured decode graph: capacity={self.capacity} window={self.max_len} "
            f"layers={len(self.layers)}"
        )

    @torch.no_grad()
    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Append one token per row and return its final hidden state (B, 1, E)."""
        if self.length >= self.max_len:
            raise RuntimeError(
                f"Window of {self.max_len} exhausted at position {self.length}."
            )
        rows = x.shape[0]
        self.write_pos.fill_(self.length)
        self.static_in[:rows].copy_(x)
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self.length += 1
        return self.static_out[:rows].clone()

    @torch.no_grad()
    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Feed `x` (B, T, E) token by token, returning the last token's hidden state."""
        out = None
        for t in range(x.shape[1]):
            out = self.step(x[:, t : t + 1])
        return out


_registry: Dict[int, StaticWindowDecoder] = {}


def decoder_for(
    module: torch.nn.Module, rows: int, max_len: int
) -> Optional[StaticWindowDecoder]:
    """Return a decoder for `module.transformer` sized to hold at least `rows` rows.

    Rows past the real batch compute unused values; every operation here is
    per-row, so padding cannot contaminate the live rows. Padding to a stable
    capacity keeps one captured graph instead of one per observed batch size.

    Returns None when the module cannot use this path, so callers fall back.
    """
    transformer = getattr(module, "transformer", None)
    if transformer is None or not torch.cuda.is_available():
        return None
    if type(transformer).__name__ != "GPT2Model":
        return None
    if getattr(transformer.config, "add_cross_attention", False):
        return None

    key = id(module)
    decoder = _registry.get(key)
    if decoder is None or decoder.capacity < rows or decoder.max_len != max_len:
        decoder = StaticWindowDecoder(transformer, max(rows, 1), max_len)
        _registry[key] = decoder
    return decoder


def reset() -> None:
    """Drop every cached decoder, e.g. after loading a different checkpoint."""
    _registry.clear()
