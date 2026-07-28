"""Timeseries data specific modules."""

from typing import List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelLoss
from .utils import find_multiple, init_weights
from ..dataset.column import SpanMeta, SpanOrigin, SpanType


class Adder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.param


class DataEncoder(nn.Module):
    """
    Encoder transforming data to embeddings with optional Perceiver-style compression.
    
    For static/condition data with many features (20-100+), use Perceiver mode
    to compress into a fixed number of latent tokens for efficient conditioning.
    """
    def __init__(
        self,
        spans: List[Tuple[int, SpanType]],
        embed_dim: int,
        need_decoder: bool = True,
        span_meta: Optional[List[SpanMeta]] = None,
       
        # Perceiver-style compression parameters
        use_perceiver: bool = False,
        num_latents: int = 16,
        perceiver_num_heads: int = 4,
        perceiver_num_layers: int = 2,
        perceiver_dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        spans : List[Tuple[int, SpanType]]
            The span types of the data. Each span will be transformed to one position in the embedded sequence.
            Its length is L, and sum of width is W.
        embed_dim : int
            The output embedded dimension (E).
        need_decoder : bool
            Whether decoder is needed.
        span_meta : Optional[List[SpanMeta]]
            Optional metadata for each span, providing information about origin (categorical, binned, etc.)
            and column name. If provided, must have same length as spans.
        use_perceiver : bool
            If True, compress all condition tokens into a fixed number of latent tokens.
            This is highly recommended for 20+ conditions. Output shape becomes [B, num_latents, E].
        num_latents : int
            Number of latent tokens for Perceiver compression (default 16).
            These tokens learn to capture all relevant condition information.
        perceiver_num_heads : int
            Number of attention heads in Perceiver cross-attention.
        perceiver_num_layers : int
            Number of Perceiver processing layers (cross-attention + self-attention).
        perceiver_dropout : float
            Dropout rate in Perceiver layers.
        """
        super().__init__()
        if span_meta is not None and len(span_meta) != len(spans):
            raise ValueError(f"span_meta length ({len(span_meta)}) must match spans length ({len(spans)})")
        
        col_encoders = []
        col_decoders = []
        for w, t in spans:
            if t == SpanType.discrete:
                col_encoders.append(nn.Embedding(w, embed_dim))
                if need_decoder:
                    col_decoders.append(Adder(w))
            elif t == SpanType.continuous:
                if w > embed_dim:
                    col_encoders.append(nn.Linear(w, embed_dim))
                else:
                    col_encoders.append(nn.ZeroPad1d((0, embed_dim - w)))
                if need_decoder:
                    col_decoders.append(nn.Linear(embed_dim, w))
            else:
                raise ValueError(f'Unsupported span type: {t}')
        self.col_encoders = nn.ModuleList(col_encoders)
        self.col_decoders = nn.ModuleList(col_decoders)
        self.spans = spans
        self.span_meta = span_meta
        self.need_decoder = need_decoder
        self.embed_dim = embed_dim
        self.num_input_tokens = len(spans)
        
        # Mask embedding for discrete spans (like [MASK] token in NLP)
        self.mask_embedding = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.mask_embedding, std=0.02)
        
        # Perceiver-style compression for many conditions
        self.use_perceiver = use_perceiver
        self.num_latents = num_latents
        if use_perceiver:
            # Learnable latent tokens that will compress all conditions
            self.latent_tokens = nn.Parameter(torch.zeros(1, num_latents, embed_dim))
            nn.init.normal_(self.latent_tokens, std=0.02)
            
            # Input normalization
            self.input_norm = nn.LayerNorm(embed_dim, eps=1e-6)
            
            # Cross-attention: latents query the condition tokens
            self.cross_attn_layers = nn.ModuleList()
            self.cross_attn_norms = nn.ModuleList()
            self.self_attn_layers = nn.ModuleList()
            
            for _ in range(perceiver_num_layers):
                # Cross-attention: latents attend to conditions
                self.cross_attn_layers.append(
                    nn.MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=perceiver_num_heads,
                        dropout=perceiver_dropout,
                        batch_first=True
                    )
                )
                self.cross_attn_norms.append(nn.LayerNorm(embed_dim, eps=1e-6))
                
                # Self-attention among latents (to mix information)
                self.self_attn_layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=perceiver_num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=perceiver_dropout,
                        activation='gelu',
                        batch_first=True,
                        norm_first=True
                    )
                )
            
            # Output projection with gating
            self.output_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )
            # Initialize output near-identity
            nn.init.zeros_(self.output_proj[-1].weight)
            nn.init.zeros_(self.output_proj[-1].bias)
        
        self.apply(init_weights)
    
    @property
    def output_num_tokens(self) -> int:
        """Number of tokens in the output (after compression if using Perceiver)."""
        if self.use_perceiver:
            return self.num_latents
        return self.num_input_tokens

    def _get_paired_mask_groups(self) -> List[List[int]]:
        """
        Get groups of span indices that should share the same mask.
        
        For binned numerics, the bin span and value span should be masked together.
        Other spans are in their own group.
        
        Returns
        -------
        List[List[int]]
            List of groups, where each group is a list of span indices that share a mask.
        """
        if self.span_meta is None:
            # No metadata, each span is its own group
            return [[i] for i in range(len(self.spans))]
        
        groups = []
        visited = set()
        
        for i, meta in enumerate(self.span_meta):
            if i in visited:
                continue
            
            if meta.paired_index is not None:
                # This span is part of a pair (binned numeric)
                group = sorted([i, meta.paired_index])
                groups.append(group)
                visited.add(i)
                visited.add(meta.paired_index)
            else:
                # Single span
                groups.append([i])
                visited.add(i)
        
        return groups

    def mask_input(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.15,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masking to input data and encode it.
        
        For categorical/discrete spans: replace with mask embedding (like [MASK] in NLP).
        For numerical/continuous spans: zero out the values.
        For binned numerics: both bin and value spans share the same mask.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data, of shape (B, W).
        mask_ratio : float
            The ratio of positions to mask (0.0 to 1.0).
        mask : Optional[torch.Tensor]
            Pre-computed mask of shape (B, num_groups). If None, will be generated.
            Each element is True if the group should be masked.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Encoded output with masking applied, shape (B, L, E) 
            - Mask tensor indicating which positions were masked, shape (B, L)
        """
        batch_size = x.shape[0]
        num_spans = len(self.spans)
        device = x.device
        
        # Get paired groups for masking
        groups = self._get_paired_mask_groups()
        num_groups = len(groups)
        
        # Generate or use provided mask (at group level)
        if mask is None:
            # Generate random mask for each group
            group_mask = torch.rand(batch_size, num_groups, device=device) < mask_ratio
        else:
            group_mask = mask
        
        # Expand group mask to span-level mask
        span_mask = torch.zeros(batch_size, num_spans, dtype=torch.bool, device=device)
        for group_idx, group in enumerate(groups):
            for span_idx in group:
                span_mask[:, span_idx] = group_mask[:, group_idx]
        
        # Encode with masking
        out = []
        st = 0
        for i, ((w, t), encoder) in enumerate(zip(self.spans, self.col_encoders)):
            span_x = x[..., st:st + w]
            
            if t == SpanType.discrete:
                ids = span_x.argmax(dim=-1)
                encoded = encoder(ids)  # (B, E)
            elif t == SpanType.continuous:
                encoded = encoder(span_x)  # (B, E)
            else:
                raise ValueError(f'Unsupported span type: {t}')
            
            # Apply masking
            mask_i = span_mask[:, i].unsqueeze(-1)  # (B, 1)
            if t == SpanType.discrete:
                # For discrete: replace with mask embedding
                encoded = torch.where(mask_i, self.mask_embedding.expand(batch_size, -1), encoded)
            else:
                # For continuous: zero out
                encoded = torch.where(mask_i, torch.zeros_like(encoded), encoded)
            
            out.append(encoded)
            st += w
        
        encoded_output = torch.stack(out, dim=-2)
        
        if self.use_perceiver:
            encoded_output = self._perceiver_compress(encoded_output)
        
        return encoded_output, span_mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input data to the embedding space.

        Parameters
        ----------
        x : torch.Tensor
            The input data, of shape (B, W).

        Returns
        -------
        torch.Tensor
            The output data:
            - If use_perceiver=True: (B, num_latents, E) - compressed latent tokens
        """
        batch_size = x.shape[0]
        
        # Step 1: Encode each feature/span to a token
        out = []
        st = 0
        for (w, t), encoder in zip(self.spans, self.col_encoders):
            span_x = x[..., st:st + w]
            if t == SpanType.discrete:
                ids = span_x.argmax(dim=-1)
                out.append(encoder(ids))
            elif t == SpanType.continuous:
                out.append(encoder(span_x))
            else:
                raise ValueError(f'Unsupported span type: {t}')
            st += w
        encoded_output = torch.stack(out, dim=-2)  # [B, num_features, E]
        
        if self.use_perceiver:
            encoded_output = self._perceiver_compress(encoded_output)
           
        return encoded_output

    def encode_one(self, span_x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Encode a single span to its token, matching `encode` for that position.

        Parameters
        ----------
        span_x : torch.Tensor
            Values of span `i` only, of shape (B, w_i).
        i : int
            The span index.

        Returns
        -------
        torch.Tensor
            The span token, of shape (B, E).
        """
        w, t = self.spans[i]
        encoder = self.col_encoders[i]
        if t == SpanType.discrete:
            return encoder(span_x.argmax(dim=-1))
        if t == SpanType.continuous:
            return encoder(span_x)
        raise ValueError(f'Unsupported span type: {t}')

    def decode_one(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Decode a single position's embedding to span `i` logits.

        Parameters
        ----------
        x : torch.Tensor
            The embedding of one position, of shape (B, E).
        i : int
            The span index.

        Returns
        -------
        torch.Tensor
            Logits for span `i`, of shape (B, w_i).
        """
        if not self.need_decoder:
            raise RuntimeError("The encoder without decoder need cannot run in decode mode.")
        w, t = self.spans[i]
        encoder = self.col_encoders[i]
        decoder = self.col_decoders[i]
        if t == SpanType.discrete:
            return decoder(x.matmul(encoder.weight.transpose(0, 1)))
        if t == SpanType.continuous:
            return decoder(x)
        raise ValueError(f'Unsupported span type: {t}')

    def _perceiver_compress(self, condition_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compress many condition tokens into fewer latent tokens using Perceiver architecture.
        
        This is the key to handling 20-100+ conditions efficiently:
        - Latent tokens learn to specialize (e.g., one for demographics, one for TS stats)
        - Cross-attention allows latents to selectively gather information
        - Self-attention mixes information between latents
        
        Parameters
        ----------
        condition_tokens : torch.Tensor
            Input condition tokens of shape (B, num_conditions, E)
        
        Returns
        -------
        torch.Tensor
            Compressed latent tokens of shape (B, num_latents, E)
        """
        B = condition_tokens.shape[0]
        
        # Normalize input conditions
        condition_tokens = self.input_norm(condition_tokens)
        
        # Initialize latents
        latents = self.latent_tokens.expand(B, -1, -1)  # [B, num_latents, E]
        
        # Iterative cross-attention and self-attention
        for cross_attn, cross_norm, self_attn in zip(
            self.cross_attn_layers, 
            self.cross_attn_norms, 
            self.self_attn_layers
        ):
            # Cross-attention: latents query the conditions
            # This is where latents gather information from all conditions
            cross_out, _ = cross_attn(
                query=latents,
                key=condition_tokens,
                value=condition_tokens
            )
            latents = cross_norm(latents + cross_out)  # Residual + norm
            
            # Self-attention: latents communicate with each other
            latents = self_attn(latents)
        
        # Output projection with gating (controls how much new info to add)
        gate = self.output_gate(latents)
        proj = self.output_proj(latents)
        latents = latents + gate * proj  # Gated residual
        
        return latents

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode the embedding space to the output data.

        Parameters
        ----------
        x : torch.Tensor
            The output data, of shape (B, L, embed_dim).

        Returns
        -------
        torch.Tensor
            The output data, of shape (B, W). Output are not raw values, but logits.
        """
        if not self.need_decoder:
            raise RuntimeError("The encoder without decoder need cannot run in decode mode.")
        out = []
        for (i, (w, t)), encoder, decoder in zip(enumerate(self.spans), self.col_encoders, self.col_decoders):
            span_x = x[..., i, :]
            if t == SpanType.discrete:
                logits = span_x.matmul(encoder.weight.transpose(0, 1))
                logits = decoder(logits)
            elif t == SpanType.continuous:
                logits = decoder(span_x)
            else:
                raise ValueError(f'Unsupported span type: {t}')
            out.append(logits)
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor, mode: Literal["encode", "decode"] = "encode") -> torch.Tensor:
        """
        Execute encode or decoder step.

        Parameters
        ----------
        x : torch.Tensor
            The input data.
        mode : Literal["encode", "decode"]
            Whether to run encode or decode step.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        if mode == "encode":
            return self.encode(x)
        elif mode == "decode":
            return self.decode(x)
        else:
            raise ValueError(f'Unsupported mode: {mode}')


class DataLoss(nn.Module):
    """
    Loss calculator of the data.
    """
    def __init__(self, spans: List[Tuple[int, SpanType]]):
        """
        Parameters
        ----------
        spans : List[Tuple[int, SpanType]]
            The span types of the data.
        """
        super().__init__()
        losses = []
        for w, t in spans:
            if t == SpanType.discrete:
                losses.append(nn.CrossEntropyLoss())
            elif t == SpanType.continuous:
                losses.append(nn.MSELoss())
            else:
                raise ValueError(f'Unsupported span type: {t}')
        self.losses = nn.ModuleList(losses)
        self.spans = spans

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> ModelLoss:
        """
        Compute the loss by comparing input to target.

        Parameters
        ----------
        x : torch.Tensor
            Model output logits.
        target : torch.Tensor
            The target data. It has the same shape as x, but is raw values instead of logits
            (difference is particularly in one-hot discrete spans).

        Returns
        -------
        ModelLoss
            The computed loss.
        """
        st = 0
        all_losses = []
        for (w, t), loss_fct in zip(self.spans, self.losses):
            span_x = x[..., st:st + w]
            span_tgt = target[..., st:st + w]
            if t == SpanType.discrete:
                col_loss = loss_fct(span_x, span_tgt.argmax(dim=-1))
            elif t == SpanType.continuous:
                col_loss = loss_fct(span_x, span_tgt)
            else:
                raise ValueError(f'Unsupported span type: {t}')
            st += w
            all_losses.append(col_loss)
        return ModelLoss(self.spans, all_losses)


class DataSampler(nn.Module):
    """
    Data sampler based on the computed logits.
    """
    def __init__(self, spans: List[Tuple[int, SpanType]], temperature: float = 1.0):
        """
        Parameters
        ----------
        spans : List[Tuple[int, SpanType]]
            The span types of the data.
        temperature : float
            The temperature for sampling.
        """
        super().__init__()
        self.spans = spans
        self.temperature = temperature

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Do the data sampling on a specific span index.

        Parameters
        ----------
        x : torch.Tensor
            The logits of the span.
        i : int
            The index of the span.

        Returns
        -------
        torch.Tensor
            The sampled data. The result will be one-hot for discrete spans.
        """
        w, t = self.spans[i]
        if t == SpanType.discrete:
            # probs = torch.softmax(x * self.temperature, dim=-1)
            probs = torch.softmax(x / self.temperature, dim=-1)
            sampled_ids = torch.multinomial(probs, num_samples=1).reshape(-1)
            out = F.one_hot(sampled_ids, w)
        elif t == SpanType.continuous:
            out = x
        else:
            raise ValueError(f'Unsupported span type: {t}')
        return out

    def cfg_merge(self, cond: torch.Tensor, uncond: torch.Tensor, filter_threshold: float, cfg: float) -> torch.Tensor:
        """
        Merge logits with classifier-free guidance.

        Parameters
        ----------
        cond : torch.Tensor
            The logits with conditional setting.
        uncond : torch.Tensor
            The logits with unconditional setting.
        filter_threshold : float
            Filter threshold for low probability tokens in cfg.
        cfg : float
            The guidance factor.

        Returns
        -------
        torch.Tensor
            The merged logits
        """
        cond_probs = torch.softmax(cond * self.temperature, dim=-1)
        mask = cond_probs < filter_threshold
        uncond[mask] = torch.max(
            uncond, cond - torch.max(cond, dim=-1, keepdim=True)[0] + torch.max(uncond, dim=-1, keepdim=True)[0]
        )[mask]
        return uncond + cfg * (cond - uncond)

class DataPatcher(nn.Module):
    """
    Module handling patching of time series.
    """
    def __init__(self, patch_size: int, seq_len: int):
        """
        Parameters
        ----------
        patch_size : int
            The patch size.
        seq_len : int
            The total sequence length.
        """
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len

    def patchify(
            self, ts_data: torch.Tensor, len_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Patchify the time series data.

        Parameters
        ----------
        ts_data : torch.Tensor
            The time series data (with length) (B, L, Wt).
        len_indicator : torch.Tensor
            The length indicator of timeseries data (B, L).

        Returns
        -------
        torch.Tensor
            Patched time series data (B, L / P, P * Wt).
        torch.Tensor
            The attention mask on the output patches (B, L / P).
        torch.Tensor
            The length indicator per patch (B, L / P, P)
        torch.Tensor
            The lengths of patched tiemseries (B), maximally L / P - 1.
        """
        bsz, length, dim = ts_data.shape
        len_to_pad = find_multiple(length, self.patch_size) - length
        padded_ts_data = F.pad(ts_data, (0, 0, 0, len_to_pad))
        padded_len_indicator = F.pad(len_indicator, (0, len_to_pad), value=-1)
        new_seq_len = (length + len_to_pad) // self.patch_size
        patched_ts_data = padded_ts_data.view(bsz, new_seq_len, self.patch_size * dim)  # B, L / P, P * Wt
        patched_len_indicator = padded_len_indicator.view(bsz, new_seq_len, self.patch_size)  # B, L / P, P
        attention_mask = (patched_len_indicator >= 0).any(dim=-1)  # B, L / P
        len_by_patch = attention_mask.sum(dim=-1)  # B
        return patched_ts_data, attention_mask.bool(), patched_len_indicator, len_by_patch

    def extract_patches(
            self, ts_data: torch.Tensor, len_indicator: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Extract patches from the time series data.

        Parameters
        ----------
        ts_data, len_indicator, cond
            Inputs to `LevelGenerator.forward`.

        Returns
        -------
        torch.Tensor
            Not empty indicator (true if not empty) (B).
        torch.Tensor, torch.Tensor, torch.Tensor
            Non-empty inputs without patching.
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            Same as `.patchify`, but empty batches are skipped (so B -> B').
        """
        # Skip empty rows
        not_empty: torch.Tensor = (len_indicator >= 0).any(-1)
        non_empty_ts_data = ts_data[not_empty]
        non_empty_len_indicator = len_indicator[not_empty]  # B', L
        non_empty_cond = cond[not_empty]

        # Get condition for next level
        non_empty_patches, non_empty_patches_am, non_empty_patched_len_indicator, non_empty_len = self.patchify(
            non_empty_ts_data, non_empty_len_indicator
        )
        return (
            not_empty, non_empty_ts_data, non_empty_len_indicator, non_empty_cond,
            non_empty_patches, non_empty_patches_am, non_empty_patched_len_indicator, non_empty_len
        )

    def patchify_for_next_level(
            self, not_empty: torch.Tensor, non_empty_patches: torch.Tensor, non_empty_patches_am: torch.Tensor,
            non_empty_patched_len_indicator: torch.Tensor, cond_next: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Patchify data for the next level, with empty data recovered.

        Parameters
        ----------
        not_empty, non_empty_patches, non_empty_patches_am, non_empty_patched_len_indicator
            First few outputs from `.extract_patches`.
        cond_next : torch.Tensor
            The conditions for the next level.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            The first few items of `LevelGenerator.forward` output.
        """
        patches = self._recover_empty(non_empty_patches, not_empty)  # B, L / P, P * Wt
        patched_len_indicator = self._recover_empty(non_empty_patched_len_indicator, not_empty, fill_value=-1)  # B, L / P, P
        patched_cond_next = self._recover_empty(cond_next, not_empty)  # B, L / P, Lx, E

        patches = patches.flatten(0, 1)
        patches = patches.view(patches.shape[0], self.patch_size, -1)  # B * L / P, P, Wt
        patched_len_indicator = patched_len_indicator.view(-1, self.patch_size)  # B * L / P, P
        patched_cond_next = patched_cond_next.flatten(0, 1)  # B * L / P, Lx, E
        return patches, patched_len_indicator, patched_cond_next

    @staticmethod
    def _recover_empty(non_empty_data: torch.Tensor, not_empty: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        data = torch.full(
            (not_empty.size(0), *non_empty_data.shape[1:]),
            fill_value=fill_value,
            device=non_empty_data.device,
            dtype=non_empty_data.dtype
        )
        data[not_empty] = non_empty_data
        return data
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify the timeseries data.

        Parameters
        ----------
        patches : torch.Tensor
            The patched timeseries data (with length) (B, L / P, P * Wt).

        Returns
        -------
        torch.Tensor
            The recovered timeseries data (B, L, Wt).
        """
        bsz, cur_seq_len, patch_dim = patches.shape
        patches = patches.view(bsz, cur_seq_len * self.patch_size, patch_dim // self.patch_size)
        return patches[..., :self.seq_len, :]