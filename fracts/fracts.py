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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import (
    TSData,
    TSDataCollator,
    TSDataset,
    TSDataTransformer,
    TSInferenceDataCollator,
    TSInferenceDataset,
)
from .model import FractalGen, MetricLogger, ModelLoss, NativeScaler, SmoothedValue
from .model.kv_cache import kv_cache
from .model.utils import add_weight_decay, adjust_batch_size, adjust_learning_rate
from .static import create_static_generator

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
        checkpoint_path: str = None,
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
        self.train_sequential(data, checkpoint_path, **sequential)

    def train_sequential(
        self,
        data: TSData,
        checkpoint_path: str = None,
        epochs: int = 1000,
        batch_size: int = 128,
        auto_adjust_batch_size: bool = True,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        memory_threshold: float = 0.80,
        num_workers: int = 8,
        pin_memory: bool = True,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        min_lr: float = 0.0,
        lr_schedule: Literal["cosine"] = "cosine",
        warmup_epochs: int = 10,
        grad_clip: float = 1.0,
        chunk_size: int = 100,
        save_epoch: int = 50,
        validation_split: float = 0.1,
        validation_freq: int = 50,
        early_stop_patience: int = 3,
        early_stop_delta: float = 1e-3,
    ):
        """
        Train the sequential part of FracTS model. Assumption is that the transformer is fitted.

        Parameters
        ----------
        data : TSData
            The training data.
        checkpoint_path: str
            Path to the saved checkpoint to continue training
        epochs : int
            The number of epochs to train the model.
        batch_size : int
            Batch size for training. Ignored if auto_adjust_batch_size is True.
        auto_adjust_batch_size : bool
            Whether to automatically find optimal batch size based on GPU memory availability.
            When True, searches between min_batch_size and max_batch_size, ignoring batch_size parameter.
        min_batch_size : int
            Minimum batch size to search when auto-adjusting (inclusive).
        max_batch_size : int
            Maximum batch size to search when auto-adjusting (inclusive).
        memory_threshold : float
            Maximum GPU memory utilization (0.0 to 1.0) when auto-adjusting batch size.
        num_workers, pin_memory
            Arguments to `DataLoader`.
        lr, weight_decay
            Arguments to `AdamW`.
        min_lr, lr_schedule, warmup_epochs
            Arguments to `adjust_learning_rate`.
        grad_clip : float
            Gradient clipping value.
        chunk_size: int
            Argument to `TSDataTransformer.get_timeseries`.
        save_epoch: int
            The number of epoch interval to save the checkpoint
        validation_split: float
            The ratio of validation data split from training data. If 0, no validation will be performed.
        validation_freq : int
            The frequency (in epochs) to perform validation if validation_split > 0.
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

        start_epoch = 0
        best_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.output_dir, "seq", "best_model.pt")
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.sequential_generator.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            loss_scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = checkpoint["epoch"] + 1

            best_loss = checkpoint.get("best_loss", float("inf"))
            patience_counter = checkpoint.get("patience_counter", 0)
            logger.info(
                f"Resumed from checkpoint: {checkpoint_path}, starting epoch {start_epoch}"
            )
        else:
            if checkpoint_path is not None:
                logger.warning(
                    f"Checkpoint path {checkpoint_path} does not exist. Starting from scratch."
                )

        ts_data = self.transformer.get_timeseries(
            data, chunk_size, os.path.join(self.output_dir, "seq-data")
        )

        n_samples = len(ts_data)
        n_val = int(n_samples * validation_split)
        if n_val > 0:
            n_val = max(n_val, batch_size)
            n_val = min(n_val, int(0.2 * n_samples))
            n_train = n_samples - n_val
            indices = torch.randperm(n_samples).tolist()
            train_indices, val_indices = indices[:n_train], indices[n_train:]
            train_data = [ts_data[i] for i in train_indices]
            val_data = [ts_data[i] for i in val_indices]
            logger.info(
                f"Split {n_samples} samples into {n_train} training and {n_val} validation samples."
            )
        else:
            train_data = ts_data
            val_data = None
            n_train = n_samples
            logger.info(f"Using all {n_samples} samples for training.")

        train_dataset = TSDataset(train_data)
        val_dataset = TSDataset(val_data) if val_data is not None else None

        # Auto-adjust batch size if enabled
        adjusted_batch_size = batch_size
        if auto_adjust_batch_size:
            logger.info(
                f"Auto-adjusting batch size based on GPU memory (ignoring provided batch_size={batch_size})..."
            )
            # Use the longest series sample to estimate worst-case memory usage
            try:
                longest_idx = max(
                    range(len(train_dataset.series)),
                    key=lambda i: train_dataset.series[i].shape[0],
                )
                collator = TSDataCollator(self.transformer.max_len)
                sample_batch = collator([train_dataset[longest_idx]])
            except Exception as e:
                logger.error(
                    f"Failed to build longest-series sample for batch size adjustment: {e}. "
                    "Falling back to the first sample."
                )
                collator = TSDataCollator(self.transformer.max_len)
                sample_batch = collator([train_dataset[0]])
            try:
                # Auto-adjust batch size (ignoring initial batch_size)
                adjusted_batch_size = adjust_batch_size(
                    initial_batch_size=batch_size,  # Only used for logging, not for search bounds
                    dataloader_sample=sample_batch,
                    model=self.sequential_generator,
                    device=device,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    memory_threshold=memory_threshold,
                )
                logger.info(
                    f"Optimal batch size found: {adjusted_batch_size} (searched between {min_batch_size} and {max_batch_size})"
                )
            except Exception as e:
                logger.error(f"Auto batch size adjustment failed: {e}")
                logger.info(f"Falling back to minimum batch size: {min_batch_size}")
                adjusted_batch_size = min_batch_size
            finally:
                # Clean up temporary tensors
                del sample_batch
                torch.cuda.empty_cache()
                gc.collect()

        ts_dataloader = DataLoader(
            train_dataset,
            batch_size=adjusted_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=TSDataCollator(self.transformer.max_len),
        )
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=adjusted_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=TSDataCollator(self.transformer.max_len),
            )
        else:
            val_dataloader = None

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

            # early stopping check
            current_loss = None
            if val_dataloader is None:
                current_loss = metric_logger.meters["loss"].global_avg

            log_writer.flush()

            # Validation
            if val_dataloader is not None and (epoch + 1) % validation_freq == 0:
                self.sequential_generator.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for (
                        static_ids,
                        static_data,
                        ts_data,
                        len_indicator,
                    ) in metric_logger.log_every(val_dataloader, print_freq, header):
                        static_data = static_data.to(device, non_blocking=True)
                        ts_data = ts_data.to(device, non_blocking=True)
                        len_indicator = len_indicator.to(device, non_blocking=True)

                        with torch.cuda.amp.autocast():
                            loss: ModelLoss = self.sequential_generator(
                                ts_data, len_indicator, static_data
                            )
                        loss_value = loss.loss.item()
                        if not math.isfinite(loss_value):
                            logger.warning(
                                "Loss is {}, stopping training".format(loss_value)
                            )
                            sys.exit(1)
                        val_loss += loss_value
                        val_steps += 1

                avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
                current_loss = avg_val_loss
                log_writer.add_scalar("train/val_loss", avg_val_loss, epoch)
                logger.info(f"Validation loss at epoch {epoch}: {avg_val_loss:.4f}")

            if current_loss is not None:
                logger.info(
                    f"Previous Best loss: {best_loss:.6f}, Delta: {best_loss - current_loss:.6f}"
                )
                if current_loss < best_loss - early_stop_delta:
                    best_loss = current_loss
                    patience_counter = 0
                    torch.save(
                        {
                            "model": self.sequential_generator.to(torch.device("cpu")),
                            "model_state": self.sequential_generator.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": loss_scaler.state_dict(),
                            "epoch": epoch,
                            "best_loss": best_loss,
                            "patience_counter": patience_counter,
                        },
                        best_model_path,
                    )
                    self.sequential_generator.to(device)
                    logger.info(
                        f"New best {'validation' if val_dataloader is not None else 'training'} loss: {best_loss:.6f}. Model saved."
                    )
                elif early_stop_patience > 0:
                    patience_counter += 1
                    logger.info(
                        f"No improvement. Patience: {patience_counter}/{early_stop_patience}"
                    )
                    if patience_counter >= early_stop_patience:
                        logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                        final_model_path = os.path.join(
                            self.output_dir, "seq", "model.pt"
                        )
                        torch.save(
                            {
                                "model": self.sequential_generator.to(
                                    torch.device("cpu")
                                ),
                                "model_state": self.sequential_generator.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": loss_scaler.state_dict(),
                                "epoch": epoch,
                                "best_loss": best_loss,
                                "patience_counter": patience_counter,
                            },
                            final_model_path,
                        )
                        break

            if (save_epoch and (epoch + 1) % save_epoch == 0) or (epoch == epochs - 1):
                checkpoint_path = os.path.join(
                    self.output_dir,
                    "seq",
                    f"checkpoint-{epoch + 1}.pt" if epoch < epochs - 1 else "model.pt",
                )
                torch.save(
                    {
                        "model": self.sequential_generator.to(torch.device("cpu")),
                        "model_state": self.sequential_generator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": loss_scaler.state_dict(),
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "patience_counter": patience_counter,
                    },
                    checkpoint_path,
                )
                self.sequential_generator.to(device)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time: {total_time_str}")

    def sample(
        self,
        n_samples: int,
        out_cache_dir: str,
        static_file: Optional[str] = None,
        static_id: Optional[str] = None,
        checkpoint_path: str = None,
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
        static_file : str, optional
            Path to the static data file. If not provided, static data will be generated.
        static_id : str, optional
            The column name of the ID column in static_file. Required if static_file is provided.
        checkpoint_path: str
            Path to the saved checkpoint to load the model from
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
        return self.sample_sequential(
            static, out_cache_dir, agg, checkpoint_path, **sequential
        )

    @torch.no_grad()
    def sample_sequential(
        self,
        static: pd.DataFrame,
        out_cache_dir: str,
        agg: Optional[pd.DataFrame] = None,
        checkpoint_path: str = None,
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
        checkpoint_path: str
            Path to the saved checkpoint to load the model from
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
        if ".length.val" in agg.columns:
            agg[".length.val"] = agg[".length.val"].round().astype(int)
        cond = self.transformer.get_norm_ts_cond(static, agg)
        lengths = None
        if not use_stop_token:
            if ".length.val" not in cond.columns:
                raise ValueError(
                    "Please provide .length.val column in cond to indicate the lengths of timeseries or set use_stop_token to True but it may not work well on variable-length timeseries."
                )
            lengths = agg[".length.val"].values.tolist()

        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.output_dir, "seq", "best_model.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.sequential_generator = checkpoint["model"].to(device)
        logger.info(f"Loaded model from {checkpoint_path}")

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
        # Reuse attention state between steps instead of re-running the whole
        # prefix. Verified to match the recomputing path to fp32 rounding.
        with kv_cache():
            for sids, cond, lengths in dataloader:
                cond = cond.to(device)
                gen, ts_len = self.sequential_generator.sample(
                    cond, num_iter_list=num_iter_list, lengths=lengths, **kwargs
                )
                ts_len += 1
                for sid, g, l in zip(sids, gen, ts_len):
                    g = g[:l].detach().cpu().numpy()
                    pd.DataFrame(
                        g, columns=self.transformer.transformed_columns
                    ).to_csv(
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
