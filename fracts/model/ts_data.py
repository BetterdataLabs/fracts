"""Timeseries data specific modules."""

from typing import List, Literal, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelLoss
from .utils import find_multiple, init_weights
from ..dataset.column import SpanType


class Adder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.param


class DataEncoder(nn.Module):
    """
    Encoder transforming data to embeddings.
    """
    def __init__(self, spans: List[Tuple[int, SpanType]], embed_dim: int, need_decoder: bool = True):
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
        """
        super().__init__()
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
        self.need_decoder = need_decoder
        self.apply(init_weights)

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
            The output data, of shape (B, L, E).
        """
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
        return torch.stack(out, dim=-2)

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