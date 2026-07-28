import logging
import math
from typing import Callable, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GPT2Config, GPT2Model, LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from ..dataset.column import SpanType
from . import kv_cache
from .config import HighLevelGenerator
from .ts_data import DataPatcher
from .utils import find_multiple, get_sinusoidal_pos_embed, init_weights

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AR(nn.Module, HighLevelGenerator):
    """
    AR model for timeseries.
    
    Uses Perceiver-compressed condition tokens as prefix in the sequence.
    The transformer naturally attends to these condition tokens.
    """

    def __init__(
        self,
        ts_width: int,
        prefix_len: int,
        seq_len: int,
        patch_size: int,
        cond_embed_dim: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        spans: List[Tuple[int, SpanType]],
        context_list: Tuple[int, ...] = (),
        grad_checkpointing: bool = False,
        use_global_cond: bool = False,
        learnable_pos_embed: bool = False,
        use_padding_mask: bool = True,
        max_batch_size: int = 65535,
        transformer_model: str = "gpt2",
        label_smoothing: float = 0.05, 
        neg_sample_ratio: int = 1,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.grad_checkpointing = grad_checkpointing
        self.use_global_cond = use_global_cond
        self.use_padding_mask = use_padding_mask
        self.ts_width = ts_width
        self.max_batch_size = max_batch_size
        self.patch_emb = nn.Linear(ts_width * patch_size, embed_dim)
        if embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by 2 for AR model."
            )
        self.patch_emb_ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.cur_seq_len = find_multiple(seq_len, patch_size) // patch_size
        self.context_list = [c for c in sorted(context_list) if c < self.cur_seq_len]
        self.pos_embed_sin = nn.Parameter(
            get_sinusoidal_pos_embed(prefix_len + self.cur_seq_len, embed_dim),
            requires_grad=learnable_pos_embed,
        )

        # Only project conditions if dimensions differ
        self.need_cond_proj = (cond_embed_dim != embed_dim)
        if self.need_cond_proj:
            self.cond_emb = nn.Linear(cond_embed_dim, embed_dim)
            logger.info(f"AR: projecting conditions {cond_embed_dim} -> {embed_dim}")

        self.max_seq_len = self.cur_seq_len + self.prefix_len
        logger.info(
            f"AR level to handle sequence length ({seq_len}): prefix_len={self.prefix_len}, "
            f"core_len={self.cur_seq_len}, patch_size={self.patch_size}, transformer_mode={transformer_model}"
        )

        if transformer_model == "gpt2":
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
            )
            self.transformer = GPT2Model(config)
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        elif transformer_model == "llama":
            config = LlamaConfig(
                hidden_size=embed_dim,
                num_hidden_layers=num_blocks,
                num_attention_heads=num_heads,
                intermediate_size=int(embed_dim * 4),
                max_position_embeddings=self.max_seq_len,
                use_cache=False,
                pad_token_id=None,
                bos_token_id=None,
                eos_token_id=None,
                attention_dropout=0.1,
            )
            self.transformer = LlamaModel(config)
            self.norm = LlamaRMSNorm(embed_dim, eps=1e-6)
        else:
            raise ValueError(f"Unsupported transformer model {transformer_model}.")

        self.len_indicator_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
        )

        self.label_smoothing = label_smoothing
        self.neg_sample_ratio = neg_sample_ratio
        
        self.data_patcher = DataPatcher(self.patch_size, self.seq_len)
        self.cond_compressor = nn.Sequential(
            nn.Linear(self.prefix_len * cond_embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim, eps=1e-6),
        )

        self.apply(init_weights)

    def predict(
        self,
        ts_data: torch.Tensor,
        attention_mask: torch.Tensor,
        cond: torch.Tensor,
        input_pos: Optional[int] = None,
        kv_caches: Optional[List] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next level's conditions.

        Parameters
        ----------
        ts_data : torch.Tensor
            Similarly to FractalGen's input, but patched. Shape is (B, L / P, P * Wt)
        attention_mask : torch.Tensor
            Attention mask on the output patches (B, L / P).
        cond : torch.Tensor
            Replacement of cond_list as a combined condition (B, Lc, Ep).
        input_pos : int, optional
            Similarly to image fractal generative model's input.
        kv_caches : list, optional
            Per-batch-slice transformer caches, updated in place. When given, only the
            token that became available at `input_pos` is fed to the transformer.

        Returns
        -------
        torch.Tensor
            The next level's conditions (B, L / P, Lx, E).
        torch.Tensor
            Length indicator predicted (B, L / P).
        """
        next_cond = []
        len_indicator = []
        for slice_idx, i in enumerate(range(0, ts_data.shape[0], self.max_batch_size)):
            this_slice = slice(i, i + self.max_batch_size)
            past = None
            if kv_caches is not None:
                while len(kv_caches) <= slice_idx:
                    kv_caches.append(None)
                past = kv_caches[slice_idx]
            this_next_cond, this_len_indicator, past = self._predict(
                ts_data[this_slice],
                attention_mask[this_slice],
                cond[this_slice],
                input_pos,
                past,
                kv_caches is not None,
            )
            if kv_caches is not None:
                kv_caches[slice_idx] = past
            next_cond.append(this_next_cond)
            len_indicator.append(this_len_indicator)
        return torch.cat(next_cond, dim=0), torch.cat(len_indicator, dim=0)

    def _predict(
        self,
        ts_data: torch.Tensor,
        attention_mask: torch.Tensor,
        cond: torch.Tensor,
        input_pos: Optional[int] = None,
        past_key_values=None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[object]]:
        incremental = use_kv_cache and input_pos is not None
        # Checkpointing only trades compute for activation memory, so it is pure
        # overhead while sampling under `no_grad`.
        use_ckpt = self.grad_checkpointing and not kv_cache.is_enabled()

        if incremental:
            emb, past_key_values = self._run_cached_step(
                ts_data, cond, input_pos, past_key_values
            )
            next_emb = emb[:, -1]  # B, E
        else:
            past_key_values = None
            # Embed time series patches
            if use_ckpt:
                emb_ts = checkpoint(self.patch_emb, ts_data)  # B, L / P, E
            else:
                emb_ts = self.patch_emb(ts_data)

            cond_emb = self._embed_cond(cond)

            emb = torch.cat([cond_emb, emb_ts], dim=1)
            emb = emb + self.pos_embed_sin[:, : emb.shape[1]]  # B, L / P + Lc, E

            # Prepare transformer inputs
            if use_ckpt:
                emb = checkpoint(self.patch_emb_ln, emb)  # B, L / P + Lc, E
            else:
                emb = self.patch_emb_ln(emb)

            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        self.prefix_len,
                        dtype=torch.bool,
                        device=ts_data.device,
                    ),
                    attention_mask,
                ],
                dim=-1,
            )

            if not self.use_padding_mask:
                attention_mask = torch.ones_like(attention_mask, dtype=torch.bool)

            if input_pos is not None:
                global_input_pos = input_pos + self.prefix_len - 1
                end_pos = self.prefix_len if input_pos == 0 else (global_input_pos + 1)
                emb = emb[:, :end_pos]
                mask = None
            else:
                mask = attention_mask.long()

            # Run through transformer (Perceiver-compressed conditions are prefix tokens)
            if use_ckpt:
                self.transformer.gradient_checkpointing_enable()
            else:
                self.transformer.gradient_checkpointing_disable()
            emb = self.transformer(
                inputs_embeds=emb,
                attention_mask=mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state

            if use_ckpt:
                emb = checkpoint(self.norm, emb)  # B, L / P + Lc, E
            else:
                emb = self.norm(emb)

            if input_pos is not None:
                next_emb = (
                    emb[:, self.prefix_len - 1] if input_pos == 0 else emb[:, -1]
                )  # B, E

        # Construct context for next level
        if input_pos is not None:
            pos_emb_for_len = self.pos_embed_sin[
                :, self.prefix_len - 1 + input_pos
            ]  # 1, E
            ctx_emb = next_emb
            next_emb_with_pos = torch.cat(
                [next_emb, pos_emb_for_len.expand(next_emb.size(0), -1)], dim=-1
            )
        else:
            next_emb = emb[:, self.prefix_len - 1 : -1]  # B, L / P, E
            pos_emb_for_len = self.pos_embed_sin[
                :, self.prefix_len - 1 : -1
            ]  # 1, L / P, E
            next_emb_with_pos = torch.cat(
                [next_emb, pos_emb_for_len.expand(next_emb.size(0), -1, -1)], dim=-1
            )  # B, L / P, 2E
            ctx_emb = [next_emb]
            for c in self.context_list:
                shifted_emb = torch.cat(
                    [
                        torch.zeros(
                            *next_emb.shape[:-2],
                            c,
                            next_emb.shape[-1],
                            dtype=next_emb.dtype,
                            device=next_emb.device,
                        ),
                        next_emb[..., :-c, :],
                    ],
                    dim=-2,
                )  # B, L / P, E
                ctx_emb.append(shifted_emb)
            ctx_emb = torch.stack(ctx_emb, dim=-2)  # B, L / P, Lx, E

        # Predict length
        cond_expanded = cond.view(cond.size(0), 1, -1)  # B, 1, Lc * Ec
        if input_pos is not None:
            cond_expanded = cond_expanded.squeeze(1)  # B, Lc * Ec
        else:
            cond_expanded = cond_expanded.expand(
                -1, next_emb_with_pos.size(1), -1
            )  # B, L / P, Lc * Ec
        if use_ckpt:
            cond_compress = checkpoint(
                self.cond_compressor, cond_expanded
            )  # B, (L / P,) E / 2
        else:
            cond_compress = self.cond_compressor(cond_expanded)

        if input_pos is None and self.use_global_cond:
            ctx_emb = torch.cat(
                [cond_compress.unsqueeze(-2), ctx_emb], dim=-2
            )  # B, L / P, Lx + 1, E
        elif input_pos == 0 and self.use_global_cond:
            ctx_emb = torch.stack([cond_compress, ctx_emb])  # 2, B, E
        combined = torch.cat([next_emb_with_pos, cond_compress], dim=-1)  # B, L / P, 3E
        if use_ckpt:
            pred_len_indicator = checkpoint(
                self.len_indicator_predictor, combined
            )  # B, (L / P,) 1
        else:
            pred_len_indicator = self.len_indicator_predictor(combined)
        return ctx_emb, pred_len_indicator[..., -1], past_key_values

    def _embed_cond(self, cond: torch.Tensor) -> torch.Tensor:
        # Project conditions only if dimensions differ
        if self.need_cond_proj:
            if self.grad_checkpointing:
                cond_emb = checkpoint(self.cond_emb, cond)  # B, Lc, E
            else:
                cond_emb = self.cond_emb(cond)
        else:
            cond_emb = cond  # Already at correct dimension

        if cond_emb.shape[1] != self.prefix_len:
            raise RuntimeError(
                f"Condition length {cond_emb.shape[1]} does not match with prefix length {self.prefix_len}."
            )
        return cond_emb

    def _run_cached_step(
        self,
        ts_data: torch.Tensor,
        cond: torch.Tensor,
        input_pos: int,
        past_key_values,
    ) -> Tuple[torch.Tensor, object]:
        """
        Run the transformer on only the tokens that are new at `input_pos`.

        At `input_pos == 0` that is the whole condition prefix, afterwards it is the
        single patch produced by the previous step. Attention over the earlier tokens
        comes from `past_key_values`, so the result matches feeding the full prefix.
        """
        if input_pos == 0:
            step_emb = self._embed_cond(cond)  # B, Lc, E
            pos_start = 0
        else:
            step_emb = self.patch_emb(ts_data[:, input_pos - 1 : input_pos])  # B, 1, E
            pos_start = self.prefix_len + input_pos - 1
        step_emb = step_emb + self.pos_embed_sin[
            :, pos_start : pos_start + step_emb.shape[1]
        ]
        step_emb = self.patch_emb_ln(step_emb)

        # Gradient checkpointing silently disables the cache inside transformers.
        self.transformer.gradient_checkpointing_disable()
        outputs = self.transformer(
            inputs_embeds=step_emb,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return self.norm(outputs.last_hidden_state), outputs.past_key_values

    def forward(
        self, ts_data: torch.Tensor, len_indicator: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            not_empty,
            non_empty_ts_data,
            non_empty_len_indicator,
            non_empty_cond,
            non_empty_patches,
            non_empty_patches_am,
            non_empty_patched_len_indicator,
            non_empty_len,
        ) = self.data_patcher.extract_patches(ts_data, len_indicator, cond)
        cond_next, pred_len_indicator = self.predict(
            non_empty_patches, non_empty_patches_am, non_empty_cond
        )  # B', L / P, Lx, E | B', L / P

        target_len_indicator = (non_empty_patched_len_indicator == 1).any(dim=-1)
        filter_len_indicator = (non_empty_patched_len_indicator == -1).all(dim=-1)

        target = target_len_indicator[~filter_len_indicator].float()
        pred = pred_len_indicator[~filter_len_indicator].squeeze(-1).view(-1)

        if target.numel() > 0:
            len_loss = self._compute_balanced_focal_loss(pred, target)
        else:
            len_loss = torch.tensor(0.0, device=ts_data.device)

        patches_wo_length, patched_len_indicator, patched_cond_next = (
            self.data_patcher.patchify_for_next_level(
                not_empty,
                non_empty_patches,
                non_empty_patches_am,
                non_empty_patched_len_indicator,
                cond_next,
            )
        )
        return (
            patches_wo_length,
            patched_len_indicator,
            patched_cond_next,
            torch.tensor(0.0, device=ts_data.device),
            len_loss,
        )
    
    def _compute_balanced_focal_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute STABLE weighted loss on ALL samples (no sampling).
        
        The previous sampling approach caused high variance/oscillation.
        This uses all samples with proper weighting for stability.
        """
        if target.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        
        num_pos = target.sum()
        num_neg = target.numel() - num_pos
        
        if num_pos == 0:
            return torch.tensor(0.0, device=pred.device)
        
        pos_weight = (num_neg / num_pos).clamp(min=1.0, max=20.0)
        
        smooth_target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        loss = F.binary_cross_entropy_with_logits(
            pred, 
            smooth_target,
            pos_weight=pos_weight,
            reduction='mean'
        )
        
        return loss

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        bsz, cur_seq_len, patch_dim = patches.shape
        patches = patches.view(
            bsz, cur_seq_len * self.patch_size, patch_dim // self.patch_size
        )
        return patches[..., : self.seq_len, :]

    def sample(
        self,
        cond: torch.Tensor,
        orig_lengths: List[int],
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
        device = cond.device
        max_bsz = cond.size(0)
        if cfg == 1.0:
            bsz = max_bsz
        else:
            bsz = max_bsz // 2

        if orig_lengths:
            seq_lengths = torch.zeros(bsz, device=device, dtype=torch.long)
            for idx, orig_len in enumerate(orig_lengths):
                num_patches = int(np.ceil(orig_len / self.patch_size))
                if num_patches > self.cur_seq_len:
                    num_patches = self.cur_seq_len
                seq_lengths[idx] = num_patches
            max_steps = seq_lengths.max().item()
            next_level_lengths = []
            for idx, orig_len in enumerate(orig_lengths):
                remaining = orig_len
                patch_lengths_for_sample = []
                for patch_idx in range(seq_lengths[idx]):
                    if remaining >= self.patch_size:
                        patch_lengths_for_sample.append(self.patch_size)
                        remaining -= self.patch_size
                    else:
                        remaining = math.ceil(remaining)
                        patch_lengths_for_sample.append(remaining)
                        remaining = 0
                next_level_lengths.append(patch_lengths_for_sample)
        else:
            max_steps = self.cur_seq_len
            seq_lengths = None
            next_level_lengths = None

        if max_steps + self.prefix_len > self.pos_embed_sin.shape[1]:
            raise RuntimeError(
                f"Cannot handle {max_steps} steps with prefix length {self.prefix_len}, "
                f"which exceeds the trained maximum {self.pos_embed_sin.shape[1] - self.prefix_len}."
            )

        patches = torch.zeros(
            bsz, max_steps, self.patch_size * self.ts_width, device=device
        )  # B, max_steps, P * Wt
        patch_lengths = torch.zeros(
            bsz, max_steps, device=device, dtype=torch.long
        )  # B, max_steps
        prev_cond_next = []
        global_cond = None
        not_finished = torch.ones(bsz, device=device, dtype=torch.bool)  # B
        lengths = torch.zeros(bsz, device=device, dtype=torch.long) - 1  # B
        attention_mask = torch.zeros(bsz, max_steps, dtype=torch.bool, device=device)
        if incomplete_allowed is None:
            incomplete_allowed = torch.ones(bsz, device=device, dtype=torch.bool)
        kv_caches = [] if kv_cache.use_kv_cache() else None
        for step in range(max_steps):
            if seq_lengths is not None:
                active_mask = step < seq_lengths
                if not active_mask.any():
                    break
                not_finished = not_finished & active_mask

            cur_patches = patches.clone()
            if not cfg == 1.0:
                patches = torch.cat([patches, patches], dim=0)
            attention_mask[not_finished, step] = 1
            cond_next, pred_cur_len_indicator = self.predict(
                patches, attention_mask, cond, step, kv_caches
            )
            if step == 0 and self.use_global_cond:
                global_cond, cond_next = cond_next
            else:
                if step == 0:
                    global_cond = None
            other_ctx = [global_cond, cond_next]
            for c in self.context_list:
                if len(prev_cond_next) < c:
                    ctx = torch.zeros_like(cond_next)
                else:
                    ctx = prev_cond_next[-c]
                other_ctx.append(ctx)
            prev_cond_next.append(cond_next)
            other_ctx = [ctx for ctx in other_ctx if ctx is not None]
            cond_next = torch.stack(other_ctx, dim=1)

            if seq_lengths is not None:
                new_finished = step >= (seq_lengths - 1)
            else:
                new_finished = torch.stack(
                    [-pred_cur_len_indicator, pred_cur_len_indicator]
                ).transpose(0, 1)
                new_finished[new_finished.isnan()] = 0
                new_finished = (
                    torch.multinomial(
                        torch.softmax(new_finished / len_temperature, -1), num_samples=1
                    )
                    .squeeze(-1)
                    .bool()
                )
                new_finished = new_finished & incomplete_allowed
                if step == max_steps - 1:
                    new_finished |= incomplete_allowed

            lengths[not_finished & new_finished] = step
            if cfg_schedule == "constant":
                cfg_iter = cfg
            elif cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (step + 1) / self.cur_seq_len
            else:
                raise ValueError(f"Invalid cfg schedule {cfg_schedule}.")
            if next_level_lengths:
                next_level_len = [
                    next_level_lengths[i][step]
                    for i in range(len(orig_lengths))
                    if not_finished[i]
                ]
            else:
                next_level_len = None
            sampled_patches, sampled_len = next_level_sample_function(
                cond=cond_next[not_finished],
                lengths=next_level_len,
                cfg=cfg_iter,
                temperature=temperature,
                filter_threshold=filter_threshold,
                incomplete_allowed=new_finished[not_finished],
            )
            sampled_patches = sampled_patches.contiguous().view(
                sampled_patches.size(0), -1
            )  # B, P * Wt
            expected = self.patch_size * self.ts_width
            if sampled_patches.size(1) != expected:
                if seq_lengths is None:
                    raise RuntimeError(
                        f"Sampled patches has wrong shape {sampled_patches.shape} at step {step}, expected {expected}."
                    )
                padded = torch.zeros(
                    sampled_patches.size(0),
                    expected,
                    device=sampled_patches.device,
                    dtype=sampled_patches.dtype,
                )
                padded[:, : min(sampled_patches.size(1), expected)] = sampled_patches[
                    :, : min(sampled_patches.size(1), expected)
                ]
                sampled_patches = padded
            cur_patches[not_finished, step] = sampled_patches.to(cur_patches.dtype)
            patches = cur_patches.clone()
            patch_lengths[not_finished, step] = sampled_len
            not_finished &= ~new_finished
            if not not_finished.any():
                break

        # B, max L / P - 1
        if seq_lengths is not None:
            lengths[lengths < 0] = seq_lengths[lengths < 0] - 1
        else:
            lengths[lengths < 0] = max_steps - 1
        actual_len = (
            lengths * self.patch_size
            + patch_lengths[torch.arange(patch_lengths.size(0)), lengths]
        )
        actual_len.clamp_(min=0, max=self.seq_len - 1)
        # B, max L - 1
        data = self.data_patcher.unpatchify(patches)  # B, L, Wt

        return data, actual_len

    @property
    def n_ctx_len(self) -> int:
        return 1 + len(self.context_list)
