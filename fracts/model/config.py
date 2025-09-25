from typing import Callable, Dict, List, Literal, Optional, Protocol, Tuple

import torch

from ..dataset.column import SpanType


class ModelLoss:
    """The model loss structure."""

    def __init__(
        self,
        spans: List[Tuple[int, SpanType]],
        losses: List[torch.Tensor],
    ):
        """
        Parameters
        ----------
        spans : List[Tuple[int, SpanType]]
            The span types of the data.
        losses : List[torch.Tensor]
            The loss computed for each span.
        """
        self.spans = spans
        self.span_losses = losses
        self.guiding_losses = {}
        self.length_losses = {}
        self.guiding_weights = {}
        self.length_weights = {}

    @property
    def loss(self) -> torch.Tensor:
        """
        The sum of loss.
        """
        data_loss = torch.stack(self.span_losses).mean()
        for k in self.guiding_losses:
            gl = self.guiding_losses[k]
            gw = self.guiding_weights[k]
            ll = self.length_losses[k]
            lw = self.length_weights[k]
            data_loss += gl * gw + ll * lw
        return data_loss

    @classmethod
    def combine(cls, losses: List["ModelLoss"]) -> "ModelLoss":
        """
        Combine multiple losses into one.

        Parameters
        ----------
        losses : List[ModelLoss]
            The list of ModelLoss to combine.

        Returns
        -------
        ModelLoss
            The combined loss.
        """
        spans = losses[0].spans
        combined_span_losses = [
            torch.stack([l.span_losses[i] for l in losses]).mean(dim=0)
            for i in range(len(spans))
        ]
        combined_loss = cls(spans, combined_span_losses)
        all_levels = set()
        for l in losses:
            all_levels.update(l.guiding_losses.keys())

        for level in all_levels:
            guiding_loss = torch.stack(
                [l.guiding_losses[level] for l in losses if level in l.guiding_losses]
            ).mean()
            length_loss = torch.stack(
                [l.length_losses[level] for l in losses if level in l.length_losses]
            ).mean()
            guiding_weight = losses[0].guiding_weights.get(level, 1.0)
            length_weight = losses[0].length_weights.get(level, 1.0)
            combined_loss.update_level(
                level, guiding_loss, length_loss, guiding_weight, length_weight
            )

        return combined_loss

    def update_level(
        self,
        level: int,
        guiding_loss: torch.Tensor,
        length_loss: torch.Tensor,
        guiding_loss_weight: float = 1.0,
        length_loss_weight: float = 1.0,
    ):
        """
        Update the losses for a level.

        Parameters
        ----------
        level : int
            The level index.
        guiding_loss : torch.Tensor
            The guiding loss value.
        length_loss : torch.Tensor
            The length loss value.
        guiding_loss_weight : float
            The weight on guiding loss.
        length_loss_weight : float
            The weight on length loss.
        """
        self.guiding_losses[level] = guiding_loss
        self.length_losses[level] = length_loss
        self.guiding_weights[level] = guiding_loss_weight
        self.length_weights[level] = length_loss_weight

    def to_dict(self) -> Dict[str, float]:
        """
        Show different aspects of the losses.

        Returns
        -------
        Dict[str, float]
            The components and loss values.
        """
        discrete = []
        continuous = []
        for (w, t), l in zip(self.spans, self.span_losses):
            if t == SpanType.discrete:
                discrete.append(l)
            else:
                continuous.append(l)
        return {
            "total": self.loss.item(),
            "data": torch.stack(self.span_losses).mean().item(),
            "discrete": (
                torch.stack(discrete).mean().item() if len(discrete) > 0 else 0.0
            ),
            "continuous": (
                torch.stack(continuous).mean().item() if len(continuous) > 0 else 0.0
            ),
            "guiding": sum(self.guiding_losses.values()).item(),
            "length": sum(self.length_losses.values()).item(),
        }


class Generator(Protocol):

    def sample(
        self,
        cond: torch.Tensor,
        incomplete_allowed: Optional[torch.Tensor] = None,
        next_level_sample_function: Optional[
            Callable[[...], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        num_iter: int = -1,
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
            Replacement of cond_list (B, Lc, Ep).
        incomplete_allowed : torch.Tensor
            Whether incomplete length (non-max length) is allowed (B).
        next_level_sample_function, num_iter, cfg, cfg_schedule, temperature, filter_threshold
            Similarly to fractal generative model for images (both AR and MAR).
            For AR, cond_list should be singleton and its shape should be B, Lc, Ep.
            The next level sample function's output for timeseries include a second item for lengths.
        len_temperature : float
            The temperature to sample length.

        Returns
        -------
        torch.Tensor
            Sampled data (B, L, Wt).
        torch.Tensor
            The predicted timeseries lengths (B), maximally L - 1.
        """
        ...


class LevelGenerator(Protocol):

    def forward(
        self, ts_data: torch.Tensor, len_indicator: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next level's conditions and compute losses.

        Parameters
        ----------
        ts_data, len_indicator, cond
            Similarly to FractalGen's input before patching. Shapes are:

            - ts_data: B, L, Wt
            - len_indicator: B, L
            - cond: B, Lc, Ep

        Returns
        -------
        torch.Tensor
            The patched data (B * L / P, P, Wt).
        torch.Tensor
            The length indicators for the patched data (B * L / P, P).
        torch.Tensor
            The next level's conditions (B * L / P, Lx, E).
        torch.Tensor
            Model guiding loss, which is 0 for AR always.
        torch.Tensor
            Length indicator loss.
        """
        ...


class EndGenerator(Protocol):
    def forward(
        self,
        ts_data: torch.Tensor,
        len_indicator: torch.Tensor,
        cond: torch.Tensor,
    ) -> ModelLoss:
        """
        Forward pass to get loss recursively.

        Parameters
        ----------
        ts_data : torch.Tensor
            The timeseries values (B, L, Wt).
        len_indicator : torch.Tensor
            The timeseries length indicators (B, L).
        cond : torch.Tensor
            The static values (B, Ws), including aggregated values, in the top level,
            or conditions from previous level (B, Lc, E).

        Returns
        -------
        ModelLoss
            The loss of the forward pass.
        """
        ...


class HighLevelGenerator(LevelGenerator, Generator, Protocol):
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
        learnable_pos_embed: bool = False,
        max_batch_size: int = 65535,
    ):
        """
        Parameters
        ----------
        ts_width : int
            The number of dimensions of TS data (Wt).
        prefix_len : int
            The prefix length from conditions (Lc).
        seq_len, patch_size, cond_embed_dim, embed_dim, num_blocks, num_heads, grad_checkpointing
            Inherited arguments from AR for image fractal generative model, except that the seq_len here refers to the
            actual length in the lowest level (L, P, Ep, E).
        spans : List[Tuple[int, SpanType]],
            The timeseries data spans.
        context_list : Tuple[int, ...]
            The context for the next level. Values should be positive integers.
            For AR, it means the previous steps' condition is passed.
            For MAR, it means the previous and next steps' conditions are all passed.
        learnable_pos_embed : bool
            Whether to allow positional embedding to be learnable.
        max_batch_size : int
            Maximum batch size of processing.
        """
        ...

    @property
    def n_ctx_len(self) -> int:
        """
        Length of next level's conditions (Lx).
        """
        ...
