import datetime
import gc
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm

from .dataset import (
    TSInferenceDataCollator,
    TSInferenceDataset,
    TSData,
    TSDataCollator,
    TSDataTransformer,
    TSDataset,
)
from .model import FractalGen, MetricLogger, ModelLoss, NativeScaler, SmoothedValue
from .model.utils import add_weight_decay, adjust_learning_rate
from .static import create_static_generator

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FracTS:
    """Core FracTS class."""

    def __init__(
        self,
        output_dir: str,
        transformer: Dict[str, Any] = {},
        static: Dict[str, Any] = {},
        sequential: Dict[str, Any] = {},
    ):
        """
        Parameters
        ----------
        output_dir : str
            Output directory.
        transformer : Dict[str, Any]
            Arguments to `TSDataTransformer`.
        static : Dict[str, Any]
            Arguments to `create_static_generator`.
        sequential : Dict[str, Any]
            Arguments to `FractalGen`. With "generator_type_list" controlled by "n_ar_layers" instead, which is an
            indicating the number of AR layers. FracTS layers will be formated as (mar, ..., mar, ar, ... , ar).
            The last layer is always ar, so theoretically the value >= 1. However, we use the value <= 0 to represent
            the structure where all layers are ar.
        """
        self.output_dir = output_dir
        self.transformer = TSDataTransformer(**transformer)
        self.static_generator = create_static_generator(
            **static, output_dir=os.path.join(self.output_dir, "static")
        )
        self._seq_args = sequential
        self.sequential_generator: Optional[FractalGen] = None

    def train(
        self,
        data: TSData,
        checkpoint_dir: str = None,
        static: Dict[str, Any] = {},
        sequential: Dict[str, Any] = {},
    ):
        """
        Train FracTS model.

        Parameters
        ----------
        data : TSData
            The training data.
        static : Dict[str, Any]
            Arguments to `StaticGenerator.train` of the static model type.
        sequential : Dict[str, Any]
            Arguments to `.train_sequential`.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.transformer.fit(data)
        torch.save(self.transformer, os.path.join(self.output_dir, "transformer.pkl"))
        static_data, agg_data = self.transformer.get_static(data)
        self.static_generator.train(
            static_data, agg_data, *self.transformer.static_standardized_types, **static
        )
        self.train_sequential(data, checkpoint_dir, **sequential)

    def train_sequential(
        self,
        data: TSData,
        checkpoint_dir: str = None,
        epochs: int = 400,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        lr: float = 5e-5,
        weight_decay: float = 0.05,
        min_lr: float = 0.0,
        lr_schedule: Literal["cosine"] = "cosine",
        warmup_epochs: int = 40,
        grad_clip: float = 3.0,
        chunk_size: int = 100,
        save_epoch: int = 50,
    ):
        """
        Train the sequential part of FracTS model. Assumption is that the transformer is fitted.

        Parameters
        ----------
        data : TSData
            The training data.
        checkpoint_dir: str
            Path to the saved checkpoint to continue training
        epochs : int
            The number of epochs to train the model.
        batch_size, num_workers, pin_memory
            Arguments to `DataLoader`.
        lr, weight_decay
            Arguments to `AdamW`.
        min_lr, lr_schedule, warmup_epochs
            Arguments to `adjust_learning_rate`.
        grad_clip : float
            Gradient clipping value.
        chunk_size
            Argument to `TSDataTransformer.get_timeseries`.
        save_epoch
            The number of epoch interval to save the checkpoint
        """
        os.makedirs(os.path.join(self.output_dir, "seq"), exist_ok=True)
        seq_len_list = tuple(self._seq_args["seq_len_list"])
        if seq_len_list[0] < self.transformer.max_len:
            raise ValueError("Please pass in a large sequence length setting.")
        for i, sl in enumerate(seq_len_list):
            if sl < self.transformer.max_len:
                break
        top_len = int(np.ceil(self.transformer.max_len / sl)) * sl
        seq_len_list = (top_len,) + seq_len_list[i:]
        self._seq_args["seq_len_list"] = seq_len_list
        for k in self._seq_args:
            if (
                k.endswith("_list")
                and k != "seq_len_list"
                and k != "generator_type_list"
            ):
                self._seq_args[k] = self._seq_args[k][i - 1 :]
        if "n_ar_layers" in self._seq_args:
            n_ar_layers = self._seq_args["n_ar_layers"]
            del self._seq_args["n_ar_layers"]
        else:
            n_ar_layers = -1
        static_spans, agg_spans = self.transformer.static_spans
        self.sequential_generator = FractalGen(
            **self._seq_args,
            generator_type_list=(
                ("ar",) * len(seq_len_list)
                if n_ar_layers <= 0
                else ("mar",) * (len(seq_len_list) - n_ar_layers)
                + ("ar",) * n_ar_layers
            ),
            static_spans=static_spans + agg_spans,
            data_spans=self.transformer.spans,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequential_generator = self.sequential_generator.to(device)
        logger.info(
            f"Number of trainable parameters: "
            f"{sum(p.numel() for p in self.sequential_generator.parameters() if p.requires_grad) / 1e6:.2f}M"
        )
        param_groups = add_weight_decay(self.sequential_generator, weight_decay)
        optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        log_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "seq"))

        # Load checkpoint if provided
        if checkpoint_dir is not None:
            checkpoint = torch.load(checkpoint_dir, map_location="cpu")
            self.sequential_generator.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            loss_scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"]
            logger.info(
                f"Loaded checkpoint from {checkpoint_dir} at epoch {start_epoch}"
            )
        else:
            start_epoch = 0
            logger.info("No checkpoint provided, starting from scratch.")

        ts_data = self.transformer.get_timeseries(
            data, chunk_size, os.path.join(self.output_dir, "seq-data")
        )
        ts_dataset = TSDataset(ts_data)
        ts_dataloader = DataLoader(
            ts_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=TSDataCollator(self.transformer.max_len),
        )

        start_time = time.time()
        logger.info(f"Start training from epoch {start_epoch} for {epochs} epochs")
        for epoch in range(start_epoch, epochs):
            self.sequential_generator.train(True)
            metric_logger = MetricLogger(delimiter="  ")
            metric_logger.add_meter(
                "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
            )
            header = f"Epoch: [{epoch}]"
            print_freq = 20

            optimizer.zero_grad()
            logger.info(f"Log dir: {log_writer.log_dir}")
            for data_iter_step, (
                static_ids,
                static_data,
                ts_data,
                len_indicator,
            ) in enumerate(metric_logger.log_every(ts_dataloader, print_freq, header)):
                adjust_learning_rate(
                    optimizer,
                    (data_iter_step + 1) / len(ts_dataloader) + epoch,
                    lr,
                    min_lr,
                    warmup_epochs,
                    epochs,
                    lr_schedule,
                )

                static_data = static_data.to(device, non_blocking=True)
                ts_data = ts_data.to(device, non_blocking=True)
                len_indicator = len_indicator.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss: ModelLoss = self.sequential_generator(
                        ts_data, len_indicator, static_data
                    )

                loss_value = loss.loss.item()
                if not math.isfinite(loss_value):
                    logger.warning("Loss is {}, stopping training".format(loss_value))
                    torch.save(
                        self.sequential_generator.to(torch.device("cpu")),
                        os.path.join(self.output_dir, "seq", "model.pt"),
                    )
                    sys.exit(1)
                loss_scaler(
                    loss.loss,
                    optimizer,
                    clip_grad=grad_clip,
                    parameters=self.sequential_generator.parameters(),
                    update_grad=True,
                )
                optimizer.zero_grad()

                metric_logger.update(loss=loss_value)
                this_lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=this_lr)

                epoch_1000x = int((data_iter_step / len(ts_dataloader) + epoch) * 1000)
                for k, v in loss.to_dict().items():
                    log_writer.add_scalar(f"train/{k}_loss", v, epoch_1000x)
                log_writer.add_scalar("lr", this_lr, epoch_1000x)

            metric_logger.synchronize_between_processes()
            logger.info(f"Averaged stats at epoch {epoch}: {metric_logger}")

            log_writer.flush()

            if save_epoch and (epoch + 1) % save_epoch == 0:
                checkpoint_path = os.path.join(
                    self.output_dir, "seq", f"checkpoint-{epoch + 1}.pth"
                )
                torch.save(
                    {
                        "model": self.sequential_generator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": loss_scaler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                torch.save(
                    self.sequential_generator.to(torch.device("cpu")),
                    os.path.join(self.output_dir, "seq", "model.pt"),
                )
                self.sequential_generator.to(device)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time: {total_time_str}")
        torch.save(
            self.sequential_generator.to(torch.device("cpu")),
            os.path.join(self.output_dir, "seq", "model.pt"),
        )

    def sample(
        self,
        n_samples: int,
        out_cache_dir: str,
        static_file: Optional[str] = None,
        static_id: Optional[str] = None,
        static: Dict[str, Any] = {},
        sequential: Dict[str, Any] = {},
    ) -> TSData:
        """
        Sample timeseries data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        out_cache_dir : str
            The output data's cache directory.
        static : Dict[str, Any]
            Arguments to `StaticGenerator.generate` of the static model type.
        sequential : Dict[str, Any]
            Arguments to `.sample_sequential`.

        Returns
        -------
        TSData
            The generated timeseries data.
        """
        os.makedirs(out_cache_dir, exist_ok=True)
        if static_file is None:
            static, agg = self.static_generator.generate(n_samples, **static)
            agg.to_csv(os.path.join(out_cache_dir, "agg-standardized.csv"), index=False)
            static.to_csv(
                os.path.join(out_cache_dir, "static-standardized.csv"), index=False
            )
        else:
            static = pd.read_csv(static_file).drop(columns=[static_id])
            if n_samples > len(static):
                raise ValueError(
                    f"n_samples {n_samples} is larger than the provided static data size {len(static)}."
                )
            static = static[:n_samples]
            static.to_csv(
                os.path.join(out_cache_dir, "static-standardized.csv"), index=False
            )
            agg = None
        return self.sample_sequential(static, out_cache_dir, agg, **sequential)

    @torch.no_grad()
    def sample_sequential(
        self,
        static: pd.DataFrame,
        out_cache_dir: str,
        agg: Optional[pd.DataFrame] = None,
        static_kwargs: Dict[str, Any] = {},
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        chunk_size: int = 100,
        use_stop_token: bool = True,
        num_iter_list: Tuple[int, ...] = (-1, -1, -1, -1, -1),
        **kwargs,
    ) -> TSData:
        """
        Sample the sequential part of FracTS model from static and aggregated information.

        Parameters
        ----------
        static : pd.DataFrame
            The static columns' values.
        out_cache_dir : str
            The output data's cache directory.
        agg : pd.DataFrame, optional
            The aggregated information's values. If not provided, they will be generated.
            If agg is to be generated, static data should be in raw format.
            If agg is provided, both agg and static should be standardized.
        static_kwargs : Dict[str, Any]
            Arguments to static generator when agg is to be generated.
        batch_size, num_workers, pin_memory
            Arguments to `DataLoader`.
        chunk_size
            Argument to `TSDataTransformer.recover`.
        use_stop_token : bool
            Whether to use stop token to indicate the end of timeseries. If False,
            the lengths of timeseries should be provided in the cond data.
        num_iter_list, **kwargs
            Arguments to `FractalGen.sample`.

        Returns
        -------
        TSData
            The generated timeseries data.
        """

        self.transformer: TSDataTransformer = torch.load(
            os.path.join(self.output_dir, "transformer.pkl")
        )
        if agg is None:
            static = self.transformer.transform_static(static, normalize=False)
            agg = self.static_generator.generate_from_static(static, **static_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cond = self.transformer.get_norm_ts_cond(static, agg)
        lengths = None
        if not use_stop_token:
            if ".length.val" not in cond.columns:
                raise ValueError(
                    "Please provide .length.val column in cond to indicate the lengths of timeseries or set use_stop_token to True but it may not work well on variable-length timeseries."
                )
            lengths = agg[".length.val"].values.tolist()
        self.sequential_generator: FractalGen = torch.load(
            os.path.join(self.output_dir, "seq", "model.pt")
        ).to(device)
        n_levels = self.sequential_generator.num_fractal_levels
        num_iter_list = num_iter_list[-n_levels:]
        dataset = TSInferenceDataset(cond, lengths)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=TSInferenceDataCollator(),
        )
        self.sequential_generator.eval()
        os.makedirs(os.path.join(out_cache_dir, "ts-normalized"), exist_ok=True)
        cond.to_csv(
            os.path.join(out_cache_dir, "static-normalized.csv"), index_label=".id"
        )

        start_time = time.time()
        logger.info(
            f"Start generating {static.shape[0]} timeseries based on static part."
        )
        pbar = tqdm(desc="Sampling TS", total=static.shape[0])
        for sids, cond, lengths in dataloader:
            cond = cond.to(device)
            gen, ts_len = self.sequential_generator.sample(
                cond, num_iter_list=num_iter_list, lengths=lengths, **kwargs
            )
            ts_len += 1
            for sid, g, l in zip(sids, gen, ts_len):
                g = g[:l].detach().cpu().numpy()
                pd.DataFrame(g, columns=self.transformer.transformed_columns).to_csv(
                    os.path.join(out_cache_dir, "ts-normalized", f"{sid}.csv"),
                    index=False,
                )
                pbar.update(1)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Finished model inference in {total_time_str}.")
        transformed_data = TSData(
            os.path.join(out_cache_dir, "ts-normalized"),
            os.path.join(out_cache_dir, "static-normalized.csv"),
            ".id",
        )

        return self.transformer.recover(
            transformed_data, static, chunk_size, os.path.join(out_cache_dir, "final")
        )
