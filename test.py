import math
import os
import unittest
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fracts import TSData, TSDataTransformer
from fracts.dataset.column import DataType, SpanType
from fracts.model.ar import AR
from fracts.model.fractalgen import FractalGen
from fracts.model.last import LastTSFractal
from fracts.model.mar import MAR
from fracts.model.ts_data import DataEncoder, DataLoss, DataPatcher, DataSampler


def edit_dict(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict):
            base[k] = edit_dict(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def sample_data_from_spans(spans: List[Tuple[int, SpanType]], n_rows: int) -> torch.Tensor:
    data = []
    for w, t in spans:
        if t == SpanType.discrete:
            ids = torch.randint(low=0, high=w, size=(n_rows,))
            span_x = F.one_hot(ids, w)
        elif t == SpanType.continuous:
            span_x = torch.normal(0, 1, size=(n_rows, w))
        else:
            raise ValueError(f"Unsupported type {t}.")
        data.append(span_x)
    data = torch.cat(data, dim=-1)
    return data


class TSTransformerTest(unittest.TestCase):
    def setUp(self):
        data_path = os.path.join("..", "dataset", "stock-sample")
        self.data = TSData(data_path, f"{data_path}.csv", "stock")
        os.makedirs(os.path.join("outputs", "test"), exist_ok=True)

    def _test_transformer(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__base__" in config:
            with open(config["__base__"], "r") as f:
                base = yaml.safe_load(f)
            config = edit_dict(base, config)
        config = config.get("general", {}).get("transformer", {})

        transformer = TSDataTransformer(**config)
        descr = os.path.basename(config_path).split('.')[0]
        with self.subTest(f"Fit ({descr})"):
            transformer.fit(self.data)

        with self.subTest(f"Get static data ({descr})"):
            static_standardized, agg_standardized = transformer.get_static(self.data, normalize=False)
            static_normalized, agg_normalized = transformer.get_static(self.data, normalize=True)

        with self.subTest(f"Get static meta ({descr})"):
            static_standardized_types, agg_standardized_types = transformer.static_standardized_types
            static_spans, agg_spans = transformer.static_spans

        with self.subTest(f"Check static meta ({descr})"):
            self._validate_standardized_dtypes(static_standardized, static_standardized_types, "static")
            self._validate_standardized_dtypes(agg_standardized, agg_standardized_types, "aggregated")

            self._validate_spans(static_normalized, static_spans, "static normalized")
            self._validate_spans(agg_normalized, agg_spans, "aggregated normalized")

        with self.subTest(f"Normalize static ({descr})"):
            normalized = transformer.get_norm_ts_cond(static_standardized, agg_standardized)
            concat_normalized = pd.concat([static_normalized, agg_normalized], axis=1)
            self._df_equal(normalized, concat_normalized, "normalized static", "get static normalized")

        os.makedirs(os.path.join("outputs", "test", f"{descr}"), exist_ok=True)
        with self.subTest(f"Get timeseries data ({descr})"):
            ts_data = transformer.get_timeseries(
                self.data, cache_dir=os.path.join("outputs", "test", f"{descr}-transformed")
            )

        with self.subTest(f"Check static from timeseries data ({descr})"):
            self._df_equal(
                ts_data.static_data.T, pd.concat([static_normalized, agg_normalized], axis=1),
                "TS static", "raw static", allow_index_shuffle=True
            )

        with self.subTest(f"Check timeseries meta ({descr})"):
            self.assertEqual(
                static_normalized.shape[0], len(ts_data), "Number of timeseries from static and TS do not match."
            )
            spans = transformer.spans
            sid, one_ts, one_st = ts_data[0]
            static_normalized_columns = pd.concat([static_normalized, agg_normalized], axis=1).columns
            self.assertTrue(static_normalized_columns.equals(one_st.index), "One item static columns wrong.")
            self._validate_spans(one_ts, spans, "first TS item spans")
            self.assertListEqual(transformer.transformed_columns, one_ts.columns.tolist(), "first TS item columns")

            batch_ts, batch_st = ts_data.get_batch(slice(1, 10))
            batch_ts = pd.concat(batch_ts, axis=0)
            self.assertTrue(static_normalized_columns.equals(batch_st.columns), "Batch static columns wrong.")
            self._validate_spans(batch_ts, spans, "batch TS spans")
            self.assertListEqual(transformer.transformed_columns, batch_ts.columns.tolist(), "batch TS columns")

        with self.subTest(f"Recover ({descr})"):
            recovered = transformer.recover(
                ts_data, static_standardized, cache_dir=os.path.join("outputs", "test", f"{descr}-recovered")
            )

        with self.subTest(f"Check recovered data ({descr})"):
            self.assertEqual(
                recovered.static_data is None, self.data.static_data is None, "Static being None recovered wrongly."
            )
            self.assertSetEqual(
                set(recovered.static_ids), set(self.data.static_ids),
                "Recovered static IDs do not match with original."
            )
            recovered.static_ids = self.data.static_ids
            self.assertEqual(len(recovered), len(self.data), "Recovered data size is different from original.")
            if recovered.static_data is not None:
                self._df_equal(
                    recovered.static_data.T, self.data.static_data.T, "recovered static", "raw static",
                    allow_index_shuffle=True
                )
            for st in range(0, len(recovered), 50):
                recov_ts, recov_st = recovered.get_batch(slice(st, st + 50))
                raw_ts, raw_st = self.data.get_batch(slice(st, st + 50))
                self._df_equal(recov_st, raw_st, f"recovered static batch from {st}", f"raw static batch from {st}")
                recov_ts = pd.concat({k: v.reset_index(drop=True) for k, v in recov_ts.items()}, axis=0)
                raw_ts = pd.concat({k: v.reset_index(drop=True) for k, v in raw_ts.items()}, axis=0)
                self._df_equal(recov_ts, raw_ts, f"recovered TS batch from {st}", f"raw TS batch from {st}")

    def _validate_standardized_dtypes(self, data: pd.DataFrame, dtypes: Dict[str, DataType], descr: str):
        cap_descr = f"{descr[0].upper()}{descr[1:]}"
        data_types = data.dtypes
        self.assertEqual(
            data.shape[-1], len(dtypes),
            f"Standardized {descr} data shape and data types size do not match."
        )
        for c, t in dtypes.items():
            self.assertIn(c, data.columns, f"Column {c} is not found in standardized {descr} data.")
            self.assertNotEqual(
                t.name, "datetime", f"Standardized type cannot be datetime, but {descr} column {c} is datetime."
            )
            if t.name == "numeric":
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(data_types[c]), f"{cap_descr} column {c} type is not {t}."
                )

    def _validate_spans(self, df: pd.DataFrame, spans: List[Tuple[int, SpanType]], descr: str):
        self.assertEqual(
            df.shape[-1], sum(w for w, t in spans),
            f"Sum of span dimensions of {descr} is not matched."
        )
        st = 0
        for i, (w, t) in enumerate(spans):
            this_span = df.values[:, st:st + w]
            cols = df.columns[st:st + w].tolist()
            if t.name == "discrete":
                self.assertTrue(
                    np.all((this_span == 0) | (this_span == 1)),
                    f"The {i}-th span of {descr} ({cols}) is discrete, and should have only 0 or 1."
                )
                self.assertTrue(
                    np.all(this_span.sum(axis=1) == 1),
                    f"The {i}-th span of {descr} ({cols}) is discrete, and should be one-hot."
                )
            else:
                self.assertLess(
                    np.mean(np.abs(this_span) > 4), 0.02,
                    f"The {i}-th span of {descr} ({cols}) is continuous, but too many numeric values out of range."
                )
            st += w

    def _df_equal(
            self, df1: pd.DataFrame, df2: pd.DataFrame, descr1: str, descr2: str,
            allow_column_shuffle: bool = False, allow_index_shuffle: bool = False, allow_reindex: bool = False
    ):
        descr1 = f"{descr1[0].upper()}{descr1[1:]}"
        self.assertTupleEqual(df1.shape, df2.shape, f"{descr1} and {descr2} shapes do not match.")
        if allow_column_shuffle:
            self.assertSetEqual(set(df1.columns), set(df2.columns), f"{descr1} and {descr2} has different columns.")
        else:
            self.assertTrue(df1.columns.equals(df2.columns), f"{descr1} and {descr2} columns do not match.")
        if not allow_reindex:
            if allow_index_shuffle:
                self.assertSetEqual(set(df1.index), set(df2.index), f"{descr1} and {descr2} has different index.")
                df2 = df2.loc[df1.index]
            else:
                self.assertTrue(df1.index.equals(df2.index), f"{descr1} and {descr2} columns do not match.")
                df1 = df1.reset_index(drop=True)
                df2 = df2.reset_index(drop=True)

        for c in df1.columns:
            col1 = df1[c]
            col2 = df2[c]
            is_datetime = (pd.api.types.is_datetime64_dtype(col1.dtype)
                           or not pd.to_datetime(col1, errors="coerce").isna().any())
            is_numeric = (pd.api.types.is_numeric_dtype(col1.dtype)
                          or not pd.to_numeric(col1, errors="coerce").isna().any())
            if is_datetime or is_numeric:
                if is_datetime and not pd.api.types.is_datetime64_dtype(col1.dtype):
                    col1 = pd.to_datetime(col1, errors="coerce")
                    col2 = pd.to_datetime(col2, errors="coerce")
                if is_numeric and not pd.api.types.is_numeric_dtype(col1.dtype):
                    col1 = pd.to_numeric(col1, errors="coerce")
                    col2 = pd.to_numeric(col2, errors="coerce")
                min_val = min(col1.min(), col2.min())
                max_val = max(col1.max(), col2.max())
                if min_val == max_val:
                    max_val += 1
                col1 = (col1 - min_val) / (max_val - min_val)
                col2 = (col2 - min_val) / (max_val - min_val)
                self.assertLess(
                    (col1 - col2).abs().quantile(0.95), 0.005, f"{descr1} and {descr2} column {c} value too different."
                )
            else:
                self.assertTrue(col1.equals(col2), f"{descr1} and {descr2} column {c} do not match exactly.")

    def test1_default(self):
        self._test_transformer("config/default.yaml")

    def test2_default_verbose(self):
        self._test_transformer("config/default-verbose.yaml")

    def test3_no_agg(self):
        self._test_transformer("config/abl-no-agg.yaml")

    def test4_no_bins(self):
        self._test_transformer("config/abl-no-bins.yaml")

    def test5_no_dat(self):
        self._test_transformer("config/abl-no-dat.yaml")

    def test6_no_diff(self):
        self._test_transformer("config/abl-no-diff.yaml")


class DataEncoderTest(unittest.TestCase):
    def _test_encoder(
            self, spans: List[Tuple[int, SpanType]], embed_dim: int = 32, n_rows: int = 500, epochs: int = 1000,
            lr: float = 5e-2, temperature: float = 1e-9
    ):
        encoder = DataEncoder(spans, embed_dim)
        loss_fct = DataLoss(spans)
        sampler = DataSampler(spans, temperature)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = encoder.to(device)
        loss_fct = loss_fct.to(device)
        sampler = sampler.to(device)
        encoder.train()
        loss_fct.train()
        print(f"Model size: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")

        data = sample_data_from_spans(spans, n_rows)

        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=n_rows)
        pbar = tqdm(total=epochs, desc="Loss: 0.0000")
        for i in range(epochs):
            for batch, in dataloader:
                optimizer.zero_grad()
                batch = batch.to(device)
                encoded = encoder(batch, mode="encode")
                decoded = encoder(encoded, mode="decode")
                loss = loss_fct(decoded, batch)
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)

        encoder.eval()
        sampler.eval()
        encoded = encoder(data.to(device), mode="encode")
        decoded = encoder(encoded, mode="decode")
        recovered = []
        st = 0
        for i, (w, t) in enumerate(spans):
            span_dec = decoded[..., st:st + w]
            rec_span = sampler(span_dec, i)
            recovered.append(rec_span)
            st += w
        recovered = torch.cat(recovered, dim=-1)

        st = 0
        data = data.to(device)
        for i, (w, t) in enumerate(spans):
            rec_span = recovered[..., st:st + w]
            x_span = data[..., st:st + w]
            if t == SpanType.discrete:
                rec_span = rec_span.argmax(dim=-1)
                x_span = x_span.argmax(dim=-1)
                self.assertLess(
                    (x_span == rec_span).float().mean().item(), (1 - 1 / w) * 0.8,
                    f"Discrete span {i} got too many errors."
                )
            elif t == SpanType.continuous:
                self.assertLess(
                    pd.Series((rec_span - x_span).abs().view(-1).detach().cpu().numpy()).quantile(0.95), 2,
                    f"Continuous span {i} ({w}) got too large errors."
                )

    def test1_single_continuous(self):
        self._test_encoder([(10, SpanType.continuous)])

    def test2_single_discrete(self):
        self._test_encoder([(10, SpanType.discrete)])

    def test3_large_continuous(self):
        self._test_encoder([(100, SpanType.continuous)])

    def test4_large_discrete(self):
        self._test_encoder([(100, SpanType.discrete)])

    def test5_mixed(self):
        self._test_encoder([
            (10, SpanType.discrete), (5, SpanType.continuous), (1, SpanType.continuous), (16, SpanType.discrete),
            (100, SpanType.continuous), (135, SpanType.discrete)
        ])


class ModelDimensionTest(unittest.TestCase):
    @staticmethod
    def _init_data(
            seq_len: int, prefix_len: int, width: int, cond_embed_dim: int, n_rows: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = torch.randint(0, seq_len * 2, (n_rows,), dtype=torch.long, device=device)
        len_indicator = F.one_hot(lengths, seq_len * 2)[:, :seq_len]
        ts_data = torch.rand(n_rows, seq_len, width, device=device)
        for i, length in enumerate(lengths):
            ts_data[i, length + 1:] = 0
        cond = torch.rand(n_rows, prefix_len, cond_embed_dim, device=device)
        return ts_data, len_indicator, cond

    def test1_last(
            self, spans: List[Tuple[int, SpanType]] = [
                (25, SpanType.discrete), (75, SpanType.continuous), (4, SpanType.continuous), (12, SpanType.discrete)
            ], cond_embed_dim: int = 128, embed_dim: int = 64, num_blocks: int = 6, num_heads: int = 4,
            n_rows: int = 50, cond_len: int = 5
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ts_data = sample_data_from_spans(spans, n_rows).to(device)
        cond = torch.randn(n_rows, cond_len, cond_embed_dim, device=device)
        len_indicator = torch.randint(0, 1, (n_rows,), device=device)
        model = LastTSFractal(
            spans, embed_dim, cond_embed_dim, num_blocks, num_heads, prev_level_ctx_len= cond_len,
        ).to(device)
        model.train()

        with self.subTest("Predict"):
            logits = model.predict(ts_data, cond)
            self.assertTupleEqual(logits.shape, ts_data.shape, "Predicted logits shape.")

        with self.subTest("Forward"):
            loss = model(ts_data, len_indicator, cond)
            self.assertEqual(loss.loss.numel(), 1, "Forward loss.")

        model.eval()
        with self.subTest("Sample"):
            sampled, lengths = model.sample(cond)
            sampled = sampled.squeeze(1)
            self.assertTupleEqual(sampled.shape, ts_data.shape, "Sampled shape.")
            self.assertTupleEqual(lengths.shape, ts_data.shape[:-1], "Lengths shape.")
            self._validate_spans(sampled, spans)

    def _validate_spans(self, sampled: torch.Tensor, spans: List[Tuple[int, SpanType]]):
        st = 0
        for i, (w, t) in enumerate(spans):
            this_span = sampled[:, st:st + w]
            if t.name == "discrete":
                self.assertTrue(
                    ((this_span == 0) | (this_span == 1)).all().item(),
                    f"The {i}-th span is discrete, and should have only 0 or 1."
                )
                self.assertTrue(
                    (this_span.sum(axis=1) == 1).all().item(),
                    f"The {i}-th span is discrete, and should be one-hot."
                )
            st += w

    def test2_patcher(
            self, ts_width: int = 13, prefix_len: int = 4, seq_len: int = 35, patch_size: int = 4,
            cond_embed_dim: int = 128, n_rows: int = 50,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ts_data, len_indicator, cond = self._init_data(
            seq_len=seq_len, prefix_len=prefix_len, width=ts_width, cond_embed_dim=cond_embed_dim,
            n_rows=n_rows, device=device
        )
        patcher = DataPatcher(patch_size, seq_len).to(device)
        n_patches = int(math.ceil(seq_len / patch_size))

        with self.subTest("Patchify"):
            not_empty = len_indicator.sum(dim=-1) == 1
            non_empty_len_indicator = len_indicator[not_empty]
            non_empty_ts_data = ts_data[not_empty]
            patches, patches_am, patched_len_indicator, len_by_patch = patcher.patchify(
                non_empty_ts_data, non_empty_len_indicator
            )
            n_not_empty = not_empty.sum()
            self.assertTupleEqual(patches.shape, (n_not_empty, n_patches, patch_size * ts_width), "Patched data.")
            self.assertTupleEqual(patches_am.shape, (n_not_empty, n_patches), "Attention mask.")
            self.assertTupleEqual(
                patched_len_indicator.shape, (n_not_empty, n_patches, patch_size), "Length indicator."
            )
            self.assertTupleEqual(len_by_patch.shape, (n_not_empty,), "Length by patch.")
            for this_patch, this_am, this_patch_len_indicator, this_len, this_len_indicator in zip(
                    patches, patches_am, patched_len_indicator, len_by_patch, non_empty_len_indicator
            ):
                this_length = this_len_indicator.argmax().item() + 1
                for i in range(n_patches):
                    st_pos = i * patch_size
                    this_patch_step = this_patch[i]
                    this_am_step = this_am[i]
                    self.assertEqual(this_am_step.item(), st_pos < this_length, f"AM at step {i}.")
                    this_patch_ts_len_step = this_patch_len_indicator[i]
                    if i == this_len.item():
                        self.assertEqual(this_patch_ts_len_step.sum().item(), 1, "Non-zero next indicator.")
                        this_step_len = this_patch_ts_len_step.long().argmax().item() + 1
                        self.assertEqual(
                            this_step_len, min(this_length - st_pos, patch_size), "Next indicator position."
                        )
                    else:
                        self.assertEqual(this_patch_ts_len_step.sum().item(), 0, "Zero next indicator.")
                        if i > this_len.item():
                            self.assertTrue((this_patch_step == 0).all(), "Empty data.")
                        # self.assertGreater(i, this_len.item(), "AM and length.")

        with self.subTest("Unpatchify"):
            unpatches = patcher.unpatchify(patches)
            self.assertTupleEqual(unpatches.shape, non_empty_ts_data.shape, "Unpatched data shape.")
            self.assertTrue(unpatches.equal(non_empty_ts_data), "Unpatched data.")

    def test3_ar(
            self, ts_width: int = 13, prefix_len: int = 4, seq_len: int = 35, patch_size: int = 4,
            cond_embed_dim: int = 128, embed_dim: int = 64, num_blocks: int = 6, num_heads: int = 4, n_rows: int = 50,
            context_list: Tuple[int, ...] = (1, 3)
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ts_data, len_indicator, cond = self._init_data(
            seq_len=seq_len, prefix_len=prefix_len, width=ts_width, cond_embed_dim=cond_embed_dim,
            n_rows=n_rows, device=device
        )
        model = AR(
            ts_width=ts_width, prefix_len=prefix_len, seq_len=seq_len, patch_size=patch_size,
            cond_embed_dim=cond_embed_dim, embed_dim=embed_dim, num_blocks=num_blocks, num_heads=num_heads,
            context_list=context_list, spans=[]
        ).to(device)
        self._validate_higher_level(
            model, ts_data, len_indicator, cond, n_rows, embed_dim, patch_size, seq_len, ts_width
        )

    def _validate_higher_level(
            self, model: Union[AR, MAR], ts_data: torch.Tensor, len_indicator: torch.Tensor, cond: torch.Tensor,
            n_rows: int, embed_dim: int, patch_size: int, seq_len: int, ts_width: int
    ):
        model.train()
        n_patches = int(math.ceil(seq_len / patch_size))
        with self.subTest("Forward"):
            patches, patched_len_indicator, cond_next, guiding_loss, len_loss = model(ts_data, len_indicator, cond)
            exp_out_rows = n_rows * n_patches
            self.assertTupleEqual(
                patches.shape, (exp_out_rows, patch_size, ts_width), "New rows od data."
            )
            self.assertTupleEqual(
                patched_len_indicator.shape, (exp_out_rows, patch_size), "New rows of length indicators."
            )
            self.assertTupleEqual(
                cond_next.shape, (exp_out_rows, model.n_ctx_len+1, embed_dim), "Next condition list."
            )
            self.assertEqual(guiding_loss.numel(), 1, "Guiding loss shape.")
            self.assertEqual(len_loss.numel(), 1, "Length loss shape.")

    def test4_mar(
            self, ts_width: int = 13, prefix_len: int = 4, seq_len: int = 35, patch_size: int = 4,
            cond_embed_dim: int = 128, embed_dim: int = 64, num_blocks: int = 6, num_heads: int = 4, n_rows: int = 50,
            context_list: Tuple[int, ...] = (1, 3),
            spans: List[Tuple[int, SpanType]] = [
                (3, SpanType.discrete), (2, SpanType.continuous), (8, SpanType.continuous)
            ]
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ts_data, len_indicator, cond = self._init_data(
            seq_len=seq_len, prefix_len=prefix_len, width=ts_width, cond_embed_dim=cond_embed_dim,
            n_rows=n_rows, device=device
        )
        model = MAR(
            ts_width=ts_width, prefix_len=prefix_len, seq_len=seq_len, patch_size=patch_size,
            cond_embed_dim=cond_embed_dim, embed_dim=embed_dim, num_blocks=num_blocks, num_heads=num_heads,
            context_list=context_list, spans=spans
        ).to(device)
        self._validate_higher_level(
            model, ts_data, len_indicator, cond, n_rows, embed_dim, patch_size, seq_len, ts_width
        )

    def test5_ar_ar(
            self,
            seq_len_list: Tuple[int, ...] = (532, 27, 1),
            embed_dim_list: Tuple[int, ...] = (64, 32, 16),
            num_blocks_list: Tuple[int, ...] = (6, 3, 2),
            num_heads_list: Tuple[int, ...] = (4, 2, 1), n_rows: int = 50,
            static_spans: List[Tuple[int, SpanType]] = [
                (5, SpanType.discrete), (135, SpanType.discrete), (3, SpanType.continuous), (2, SpanType.discrete)
            ],
            data_spans: List[Tuple[int, SpanType]] = [
                (25, SpanType.discrete), (75, SpanType.continuous), (4, SpanType.continuous), (12, SpanType.discrete)
            ],
            context_list: Tuple[Tuple[int, ...], ...] = ((1, 3), (), ()),
            num_iter_list: Tuple[int] = (-1, -1, -1), cfg: float = 1.0,
            cfg_schedule: Literal["constant", "linear"] = "linear",
            temperature: float = 1.0, filter_threshold: float = 1e-4,
            len_temperature: float = 1.0
    ):
        generator_type_list = ("ar",) * len(seq_len_list)
        self._validate_full(
            seq_len_list, embed_dim_list, num_blocks_list, num_heads_list, generator_type_list, n_rows, static_spans,
            data_spans, context_list, num_iter_list, cfg, cfg_schedule, temperature, filter_threshold, len_temperature
        )

    def test6_mar_mar(
            self,
            seq_len_list: Tuple[int, ...] = (532, 27, 1),
            embed_dim_list: Tuple[int, ...] = (64, 32, 16),
            num_blocks_list: Tuple[int, ...] = (6, 3, 2),
            num_heads_list: Tuple[int, ...] = (4, 2, 1), n_rows: int = 50,
            static_spans: List[Tuple[int, SpanType]] = [
                (5, SpanType.discrete), (135, SpanType.discrete), (3, SpanType.continuous), (2, SpanType.discrete)
            ],
            data_spans: List[Tuple[int, SpanType]] = [
                (25, SpanType.discrete), (75, SpanType.continuous), (4, SpanType.continuous), (12, SpanType.discrete)
            ],
            context_list: Tuple[Tuple[int, ...], ...] = ((1, 3), (), ()),
            num_iter_list: Tuple[int] = (100, 100, 100), cfg: float = 1.0,
            cfg_schedule: Literal["constant", "linear"] = "linear",
            temperature: float = 1.0, filter_threshold: float = 1e-4,
            len_temperature: float = 1.0
    ):
        generator_type_list = ("mar",) * (len(seq_len_list) - 1) + ("ar",)
        self._validate_full(
            seq_len_list, embed_dim_list, num_blocks_list, num_heads_list, generator_type_list, n_rows, static_spans,
            data_spans, context_list, num_iter_list, cfg, cfg_schedule, temperature, filter_threshold, len_temperature
        )

    def test7_mar_ar(
            self,
            seq_len_list: Tuple[int, ...] = (532, 27, 1),
            embed_dim_list: Tuple[int, ...] = (64, 32, 16),
            num_blocks_list: Tuple[int, ...] = (6, 3, 2),
            num_heads_list: Tuple[int, ...] = (4, 2, 1), n_rows: int = 50, n_mar: int = 1,
            static_spans: List[Tuple[int, SpanType]] = [
                (5, SpanType.discrete), (135, SpanType.discrete), (3, SpanType.continuous), (2, SpanType.discrete)
            ],
            data_spans: List[Tuple[int, SpanType]] = [
                (25, SpanType.discrete), (75, SpanType.continuous), (4, SpanType.continuous), (12, SpanType.discrete)
            ],
            context_list: Tuple[Tuple[int, ...], ...] = ((1, 3), (), ()),
            num_iter_list: Tuple[int] = (100, 100, 100), cfg: float = 1.0,
            cfg_schedule: Literal["constant", "linear"] = "linear",
            temperature: float = 1.0, filter_threshold: float = 1e-4,
            len_temperature: float = 1.0
    ):
        generator_type_list = ("mar",) * n_mar + ("ar",) * (len(seq_len_list) - n_mar)
        self._validate_full(
            seq_len_list, embed_dim_list, num_blocks_list, num_heads_list, generator_type_list, n_rows, static_spans,
            data_spans, context_list, num_iter_list, cfg, cfg_schedule, temperature, filter_threshold, len_temperature
        )

    def _validate_full(
            self,
            seq_len_list: Tuple[int, ...] = (532, 27, 1),
            embed_dim_list: Tuple[int, ...] = (64, 32, 16),
            num_blocks_list: Tuple[int, ...] = (6, 3, 2),
            num_heads_list: Tuple[int, ...] = (4, 2, 1),
            generator_type_list: Tuple[Literal["ar", "mar"]] = ("ar", "ar", "ar"),
            n_rows: int = 50,
            static_spans: List[Tuple[int, SpanType]] = [
                (5, SpanType.discrete), (135, SpanType.discrete), (3, SpanType.continuous), (2, SpanType.discrete)
            ],
            data_spans: List[Tuple[int, SpanType]] = [
                (25, SpanType.discrete), (75, SpanType.continuous), (4, SpanType.continuous),
                (12, SpanType.discrete)
            ],
            context_list: Tuple[Tuple[int, ...], ...] = ((1, 3), (), ()),
            num_iter_list: Tuple[int] = (-1, -1, -1), cfg: float = 1.0,
            cfg_schedule: Literal["constant", "linear"] = "linear",
            temperature: float = 1.0, filter_threshold: float = 1e-4,
            len_temperature: float = 1.0
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len = seq_len_list[0]
        lengths = torch.randint(0, seq_len, (n_rows,), dtype=torch.long, device=device)
        len_indicator = F.one_hot(lengths, seq_len)
        ts_data = sample_data_from_spans(data_spans, n_rows * seq_len).to(device)
        ts_data = ts_data.view(n_rows, seq_len, *ts_data.shape[1:])
        for i, length in enumerate(lengths):
            ts_data[i, length + 1:] = 0
        static = sample_data_from_spans(static_spans, n_rows).to(device)
        model = FractalGen(
            seq_len_list=seq_len_list,
            embed_dim_list=embed_dim_list,
            num_blocks_list=num_blocks_list,
            num_heads_list=num_heads_list,
            generator_type_list=generator_type_list,
            static_spans=static_spans,
            data_spans=data_spans,
            context_list=context_list
        ).to(device)
        model.train()

        with self.subTest("Forward"):
            loss = model(ts_data, len_indicator, static)
            self.assertEqual(loss.loss.numel(), 1, "Forward loss.")

        model.eval()
        with self.subTest("Sample"):
            gen, ts_len = model.sample(
                static, num_iter_list, None, cfg, cfg_schedule, temperature, filter_threshold, len_temperature
            )
            self.assertTupleEqual(gen.shape, ts_data.shape, "Generated data shape.")
            self.assertTupleEqual(ts_len.shape, ts_data.shape[:-2], "Generated length shape.")
            self.assertTrue((ts_len < seq_len).all().item(), "Sequence length limit.")
            all_samples = []
            for g, l in zip(gen, ts_len):
                all_samples.append(g[:l + 1])
            gen = torch.cat(all_samples, dim=0)
            self._validate_spans(gen, data_spans)
