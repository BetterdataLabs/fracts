import functools
import logging
from typing import Callable, List, Literal, Optional, Tuple

import torch
from torch import nn

from ..dataset.column import SpanType
from .config import EndGenerator, Generator, ModelLoss
from .ts_data import DataEncoder, DataLoss, DataSampler
from .utils import init_weights, get_sinusoidal_pos_embed
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2Model, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LastTSFractal(nn.Module, Generator, EndGenerator):
    """Last layer, replacement for pixel loss in images."""

    def __init__(
        self,
        spans: List[Tuple[int, SpanType]],
        embed_dim: int,
        cond_embed_dim: int,
        num_blocks: int,
        num_heads: int,
        max_batch_size: int = 65535,
        prev_level_ctx_len: int = 0,
        grad_checkpointing: bool = True,
        transformer_model: str = "gpt2",
    ):
        """
        Parameters
        ----------
        spans : List[Tuple[int, SpanType]]
            Timeseries data spans. Its sum of width is Wt and length is N.
        embed_dim : int
            The embedding dimension of the last layer.
        cond_embed_dim : int
            The embedding dimension of the previous layer.
        num_blocks : int
            The number of causal blocks.
        num_heads : int
            The number of heads.
        max_batch_size : int
            Maximum batch size of processing.
        prev_level_ctx_len : int
            The previous level's context length.
        """
        super().__init__()
        self.spans = spans
        self.max_batch_size = max_batch_size
        self.seq_len = len(spans)
        self.width = sum(w for w, t in spans)
        logger.info(
            f"Last level: prefix_len={prev_level_ctx_len}, core_len={self.seq_len}, width={self.width}, transformer_mode={transformer_model}"
        )
        ends = [0]
        st = 0
        for w, t in spans:
            st += w
            ends.append(st)
        self.ends = torch.tensor(ends)
        self.data_encoder = DataEncoder(spans, embed_dim)
        self.cond_proj = nn.Linear(cond_embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pos_embedding = nn.Parameter(
            get_sinusoidal_pos_embed(prev_level_ctx_len + self.seq_len, embed_dim),
            requires_grad=False,
        )

        self.max_seq_len = prev_level_ctx_len + self.seq_len
        if transformer_model:
            config = GPT2Config(
                n_embd=embed_dim,
                n_layer=num_blocks,
                n_head=num_heads,
                n_inner=int(embed_dim * 4),
                activation_function="gelu",
                n_positions=self.max_seq_len,
                use_cache=False,
                pad_token_id=None,
                bos_token_id=None,
                eos_token_id=None,
                attn_pdrop=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
            )
            self.transformer = GPT2Model(config)
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        elif transformer_model == "llama":
            config = LlamaConfig(
                hidden_size=embed_dim,
                num_hidden_layers=num_blocks,
                num_attention_heads=num_heads,
                intermediate_size=int(embed_dim * 4),
                hidden_act="silu",
                max_position_embeddings=self.max_seq_len,
                rms_norm_eps=1e-6,
                use_cache=False,
                pad_token_id=None,
                bos_token_id=None,
                eos_token_id=None,
                attention_dropout=0.0,
                rope_scaling=None,
                rope_theta=10000.0,
            )
            self.transformer = LlamaModel(config)
            self.norm = LlamaRMSNorm(embed_dim, eps=1e-6)
        else:
            raise ValueError(f"Unsupported transformer model {transformer_model}.")

        self.loss_func = DataLoss(spans)
        self.apply(init_weights)
        self.grad_checkpointing = grad_checkpointing
        print(
            f"LastTSFractal initialized with grad_checkpointing={self.grad_checkpointing}"
        )

    def predict(self, ts_data: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Predict next level's conditions.

        Parameters
        ----------
        ts_data : torch.Tensor
            Data of one time step. Shape is (B, Wt)
        cond : torch.Tensor
            Condition (B, Lc, Ep).

        Returns
        -------
        torch.Tensor
            The predicted values (B, Wt).
        """
        if ts_data.shape[0] == 0:
            return torch.empty(
                0, self.width, device=ts_data.device, dtype=ts_data.dtype
            )
        pred = []
        for i in range(0, ts_data.shape[0], self.max_batch_size):
            this_slice = slice(i, i + self.max_batch_size)
            this_pred = self._predict(ts_data[this_slice], cond[this_slice])
            pred.append(this_pred)
        return torch.cat(pred, dim=0)

    def _predict(self, ts_data: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        if self.grad_checkpointing:
            cond = checkpoint(self.cond_proj, cond)  # B, Lc, E
        else:
            cond = self.cond_proj(cond)
        if self.grad_checkpointing:
            ts_data = checkpoint(self.data_encoder, ts_data)  # B, N, E
        else:
            ts_data = self.data_encoder(ts_data)

        data = torch.cat([cond, ts_data], dim=1) + self.pos_embedding  # B, Lc + N, E

        if self.grad_checkpointing:
            data = checkpoint(self.ln, data)  # B, Lc + N, E
        else:
            data = self.ln(data)

        if self.grad_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        else:
            self.transformer.gradient_checkpointing_disable()
        outputs = self.transformer(
            inputs_embeds=data,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        data = outputs.last_hidden_state

        if self.grad_checkpointing:
            data = checkpoint(self.norm, data)  # B, Lc + N, E
        else:
            data = self.norm(data)

        ts_data = data[..., -self.seq_len - 1 : -1, :]  # B, N, E
        with torch.cuda.amp.autocast(enabled=False):
            logits = self.data_encoder(ts_data, mode="decode")  # B, Wt

        return logits

    def forward(
        self, ts_data: torch.Tensor, len_indicator: torch.Tensor, cond: torch.Tensor
    ) -> ModelLoss:

        ts_data = ts_data.squeeze(-2)  # B, Wt
        len_indicator = len_indicator.squeeze(-1)  # B
        not_empty = len_indicator >= 0
        non_empty_ts_data = ts_data[not_empty]  # B', Wt
        non_empty_cond = cond[not_empty]

        logits = self.predict(non_empty_ts_data, non_empty_cond)  # B', Wt
        loss = self.loss_func(logits, non_empty_ts_data)  # B', Wt
        return loss

    def sample(
        self,
        cond: torch.Tensor,
        lengths: List[int],
        incomplete_allowed: Optional[torch.Tensor] = None,
        next_level_sample_function: Optional[
            Callable[..., Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        num_iter: int = -1,
        cfg: float = 1.0,
        cfg_schedule: Literal["constant", "linear"] = "linear",
        temperature: float = 1.0,
        filter_threshold: float = 1e-4,
        len_temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cfg == 1.0:
            bsz = cond.shape[0]
        else:
            bsz = cond.shape[0] // 2
        out = torch.zeros(bsz, self.width, device=cond.device)
        sampler = DataSampler(self.spans, temperature)

        for step in range(self.seq_len):
            if cfg == 1.0:
                to_pred = out
            else:
                to_pred = torch.cat([out, out], dim=0)
            logits = self.predict(to_pred, cond)
            st, ed = self.ends[step : step + 2]
            logits = logits[..., st:ed]  # B, Ws

            if not cfg == 1.0:
                cond_logits = logits[:bsz]
                uncond_logits = logits[bsz:]
                logits = sampler.cfg_merge(
                    cond_logits, uncond_logits, filter_threshold, cfg
                )

            sampled_span = sampler(logits, step)
            out[..., st:ed] = sampled_span

        return out.unsqueeze(-2), torch.zeros(bsz, device=out.device, dtype=torch.long)
