import functools
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn

from ..dataset.column import SpanType
from .ar import AR
from .config import EndGenerator, ModelLoss
from .last import LastTSFractal
from .ts_data import DataEncoder


class FractalGen(nn.Module, EndGenerator):
    """Fractal generative model adapted for TS."""

    def __init__(
        self,
        seq_len_list: Tuple[int, ...],
        embed_dim_list: Tuple[int, ...],
        num_blocks_list: Tuple[int, ...],
        num_heads_list: Tuple[int, ...],
        generator_type_list: Tuple[Literal["mar", "ar"]],
        static_spans: List[Tuple[int, SpanType]],
        data_spans: List[Tuple[int, SpanType]],
        context_list: Tuple[Tuple[int, ...], ...],
        static_drop_prob: float = 1.0,
        guiding_loss_weight: float = 1.0,
        length_loss_weight: float = 0.1,
        grad_checkpointing: bool = False,
        use_global_cond: bool = False,
        use_padding_mask: bool = True,
        transformer_model: str = "gpt2",
        fractal_level: int = 0,
        prev_level_ctx_len: int = 0,
        learnable_pos_embed: bool = False,
        max_batch_size: int = 65535,
        
        use_perceiver: bool = True, 
        num_latents: int = 16,  # Number of latent tokens (compress 100 conditions → 16 tokens)
        perceiver_num_heads: int = 8,  # Attention heads for Perceiver
        perceiver_num_layers: int = 4,  # Cross-attention + self-attention layers
    ):
        """
        Parameters
        ----------
        seq_len_list : Tuple[int, ...]
            Replacement of `img_size_list` for fractal generative model on images to sequence lengths for time series.
        embed_dim_list, num_blocks_list, num_heads_list, generator_type_list, fractal_level, grad_checkpointing
            Arguments to the original `FractalGen` module in fractal generative model.
        context_list : Tuple[Tuple[int, ...], ...]
            The context list at each level.
        static_spans : List[Tuple[int, SpanType]]
            The input static data's spans.
        data_spans : List[Tuple[int, SpanType]]
            The input timeseries data's spans.
        static_drop_prob : float
            Similarly to `label_drop_prob` in the original fractal generative model.
        guiding_loss_weight : float
            The weight on guiding loss compared to loss from previous level.
        length_loss_weight : float
            The weight on length loss compared to loss from previous level.
        prev_level_ctx_len : int
            The previous level's context length. Used for non-top levels.
        learnable_pos_embed : bool
            Argument to HigherLevelGenerator.
        max_batch_size : int
            Maximum batch size at each level. The actual batch size of the processing is the specified batch size
            multiplied by the total number of patches in the data, which can be very large in lower levels. This
            parameter aims to control the actual batch size of processing to avoid out of memory at lower levels.
        use_perceiver : bool
            If True, use Perceiver-style compression to handle many conditions efficiently.
        num_latents : int
            Number of latent tokens when using Perceiver compression.
        perceiver_num_heads : int
            Number of attention heads in Perceiver cross/self-attention.
        perceiver_num_layers : int
            Number of cross-attention + self-attention layers in Perceiver.
        """
        super().__init__()
        # Store initialization parameters for test model creation
        self._init_params = {
            "seq_len_list": seq_len_list,
            "embed_dim_list": embed_dim_list,
            "num_blocks_list": num_blocks_list,
            "num_heads_list": num_heads_list,
            "generator_type_list": generator_type_list,
            "static_spans": static_spans,
            "data_spans": data_spans,
            "context_list": context_list,
            "static_drop_prob": static_drop_prob,
            "guiding_loss_weight": guiding_loss_weight,
            "length_loss_weight": length_loss_weight,
            "grad_checkpointing": grad_checkpointing,
            "use_global_cond": use_global_cond,
            "use_padding_mask": use_padding_mask,
            "transformer_model": transformer_model,
            "fractal_level": fractal_level,
            "prev_level_ctx_len": prev_level_ctx_len,
            "learnable_pos_embed": learnable_pos_embed,
            "max_batch_size": max_batch_size,
            "use_perceiver": use_perceiver,
            "num_latents": num_latents,
            "perceiver_num_heads": perceiver_num_heads,
            "perceiver_num_layers": perceiver_num_layers,
        }

        self.fractal_level = fractal_level
        self.num_fractal_levels = len(seq_len_list)
        if fractal_level == 0:
            self.static_encoder = DataEncoder(
                static_spans, embed_dim_list[0], need_decoder=False,
                use_perceiver=use_perceiver, num_latents=num_latents,
                perceiver_num_heads=perceiver_num_heads, perceiver_num_layers=perceiver_num_layers
            )
            self.static_drop_prob = static_drop_prob
            prefix_len = self.static_encoder.output_num_tokens
        else:
            prefix_len = prev_level_ctx_len

        current_type = generator_type_list[fractal_level]
        if current_type == "ar":
            print(f"Using AR generator at fractal level {fractal_level}.")
            generator = AR
        else:
            raise NotImplementedError(
                f"Generator of type {current_type} is not supported."
            )

        self.generator = generator(
            ts_width=sum(w for w, t in data_spans),
            prefix_len=prefix_len + (1 if fractal_level > 0 and use_global_cond else 0),
            seq_len=seq_len_list[fractal_level],
            patch_size=seq_len_list[fractal_level + 1],
            cond_embed_dim=embed_dim_list[max(fractal_level - 1, 0)],
            embed_dim=embed_dim_list[fractal_level],
            context_list=context_list[fractal_level],
            num_blocks=num_blocks_list[fractal_level],
            num_heads=num_heads_list[fractal_level],
            grad_checkpointing=grad_checkpointing,
            learnable_pos_embed=learnable_pos_embed,
            max_batch_size=max_batch_size,
            spans=data_spans,
            use_global_cond=use_global_cond,
            transformer_model=transformer_model,
            use_padding_mask=use_padding_mask,
        )

        if fractal_level < self.num_fractal_levels - 2:
            self.next_fractal = FractalGen(
                seq_len_list=seq_len_list,
                embed_dim_list=embed_dim_list,
                num_blocks_list=num_blocks_list,
                num_heads_list=num_heads_list,
                generator_type_list=generator_type_list,
                context_list=context_list,
                static_spans=static_spans,
                data_spans=data_spans,
                static_drop_prob=static_drop_prob,
                fractal_level=fractal_level + 1,
                prev_level_ctx_len=self.generator.n_ctx_len,
                max_batch_size=max_batch_size,
                grad_checkpointing=grad_checkpointing,
                use_global_cond=use_global_cond,
                transformer_model=transformer_model,
                use_padding_mask=use_padding_mask,
                use_perceiver=use_perceiver,
                num_latents=num_latents,
                perceiver_num_heads=perceiver_num_heads,
                perceiver_num_layers=perceiver_num_layers,
            )
        else:
            self.next_fractal = LastTSFractal(
                spans=data_spans,
                embed_dim=embed_dim_list[fractal_level + 1],
                cond_embed_dim=embed_dim_list[fractal_level],
                num_blocks=num_blocks_list[fractal_level + 1],
                num_heads=num_heads_list[fractal_level + 1],
                max_batch_size=max_batch_size,
                prev_level_ctx_len=self.generator.n_ctx_len
                + (1 if use_global_cond else 0),
                grad_checkpointing=grad_checkpointing,
                transformer_model=transformer_model,
            )

        self.guiding_loss_weight = guiding_loss_weight
        self.length_loss_weight = length_loss_weight

    def forward(
        self,
        ts_data: torch.Tensor,
        len_indicator: torch.Tensor,
        cond: torch.Tensor,
        max_batch_size: int = 65535,
    ) -> ModelLoss:
        cond = self._process_cond(cond)
        batch_size = ts_data.shape[0]
        if batch_size <= max_batch_size:
            ts_data, len_indicator, cond, guiding_loss, len_loss = self.generator(
                ts_data, len_indicator, cond
            )
            loss: ModelLoss = self.next_fractal(ts_data, len_indicator, cond)
            loss.update_level(
                self.fractal_level,
                guiding_loss,
                len_loss,
                self.guiding_loss_weight,
                self.length_loss_weight,
            )
            return loss
        losses = []
        guiding_losses = []
        length_losses = []

        for i in range(0, batch_size, max_batch_size):

            ts_data_i = ts_data[i : i + max_batch_size]
            len_indicator_i = len_indicator[i : i + max_batch_size]
            cond_i = cond[i : i + max_batch_size]
            ts_data_i, len_indicator_i, cond_i, guiding_loss, len_loss = self.generator(
                ts_data_i, len_indicator_i, cond_i
            )
            sub_loss: ModelLoss = self.next_fractal(ts_data_i, len_indicator_i, cond_i)
            sub_loss.update_level(
                self.fractal_level,
                guiding_loss,
                len_loss,
                self.guiding_loss_weight,
                self.length_loss_weight,
            )
            losses.append(sub_loss)
            guiding_losses.append(guiding_loss)
            length_losses.append(len_loss)

        combined_loss = ModelLoss.combine(losses)
        combined_guiding_loss = torch.stack(guiding_losses).mean()
        combined_length_loss = torch.stack(length_losses).mean()
        combined_loss.update_level(
            self.fractal_level,
            combined_guiding_loss,
            combined_length_loss,
            self.guiding_loss_weight,
            self.length_loss_weight,
        )

        return combined_loss

    def _process_cond(self, cond: torch.Tensor) -> torch.Tensor:
        if self.fractal_level == 0:
            if self.training and self.static_drop_prob > 0:
                static_embedding, _ = self.static_encoder.mask_input(
                    cond, mask_ratio=self.static_drop_prob
                )
            else:
                static_embedding = self.static_encoder.encode(cond)
            cond = static_embedding
        return cond

    def sample(
        self,
        cond: torch.Tensor,
        num_iter_list: Tuple[int, ...],
        lengths: List[int],
        incomplete_allowed: Optional[torch.Tensor] = None,
        cfg: float = 1.0,
        cfg_schedule: Literal["constant", "linear"] = "linear",
        temperature: float = 1.0,
        filter_threshold: float = 1e-4,
        len_temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample data.

        Parameters
        ----------
        cond : torch.Tensor
            Replacement of cond_list (B, Lc, Ep), input `cond` of `forward`.
        incomplete_allowed : torch.Tensor
            Whether length smaller than the max length at the level is allowed (B).
        num_iter_list, cfg, cfg_schedule, temperature, filter_threshold
            Similarly to fractal generative model for images.
        len_temperature
            Argument to `HigherLevelGenerator.sample`.

        Returns
        -------
        torch.Tensor
            Sampled data (B, L, Wt).
        torch.Tensor
            The predicted timeseries lengths (B), maximally L - 1.
        """
        cond = self._process_cond(cond)
        if self.fractal_level < self.num_fractal_levels - 2:
            next_level_sample_function = functools.partial(
                self.next_fractal.sample,
                num_iter_list=num_iter_list,
                cfg_schedule="constant",
                temperature=temperature,
                filter_threshold=filter_threshold,
            )
        else:
            next_level_sample_function = self.next_fractal.sample

        return self.generator.sample(
            cond,
            lengths,
            incomplete_allowed,
            next_level_sample_function,
            num_iter_list[self.fractal_level],
            cfg,
            cfg_schedule,
            temperature,
            filter_threshold,
            len_temperature,
        )
