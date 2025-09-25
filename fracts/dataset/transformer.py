import collections
import datetime
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from .column import ColumnTransformer, DataType, SpanType, column_transformers
from .data import TSData
from .trend import TrendColumnTransformer, trend_transformers
from .utils import extract, flatten_columns

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TSDataTransformer:
    """
    Data transformer of timeseries data to prepare neural network friendly data format for both static and timeseries
    component. Accepted data should satisfy all the following constraints (including both static and timeseries data):

    - No missing data present.
    - All columns should be one of the following 3 data types except for the static ID: categorical, numeric, datetime
      (without timezone).
    - Absence of high number of categories. The number of different categories in a categorical column should be
      limited (e.g., the number in 10^5 scale is not recommended).
    - All numeric and datetime columns should be in standard formats
      (recognized by `pd.to_datetime` and `pd.to_numeric`).
    - All static and timeseries columns have different names, and column names do not contain the character ".".
    """
    def __init__(
            self, min_n_continuous: int = 20, dayfirst: bool = False, yearfirst: bool = False, agg_length: bool = True,
            st_categorical_cols: List[str] = None, st_numeric_cols: List[str] = None, ts_categorical_cols: List[str] = None,
            ts_numeric_cols: List[str] = None,  **kwargs):
        """
        Parameters
        ----------
        min_n_continuous: int
            The minimum number of unique values for a column to be considered continuous. If the number of unique values
            is smaller than the value, the column is considered to be categorical.
        dayfirst : bool
            The datetime format convention about day being first.
        yearfirst : bool
            The datetime format convention about year being first.
        agg_length : bool
            Whether to aggregate with length information.
        **kwargs
            Parameter for transformer of each column type, including trend transformer.
        """
        self.min_n_continuous = min_n_continuous
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self._column_params = {
            t: {
                k: v for k, v in kwargs.items() if k in transformer.params
            } for t, transformer in column_transformers.items()
        }
        self._trend_params = {
            t: {
                k: v for k, v in kwargs.items() if k in transformer.params
            } for t, transformer in trend_transformers.items()
        }

        self.st_num_cols = []
        self.st_cat_cols = []
        if st_categorical_cols is not None:
            self.st_cat_cols.extend(st_categorical_cols)
        if st_numeric_cols is not None:
            self.st_num_cols.extend(st_numeric_cols)
        self.st_dat_cols = []
        self.ts_num_cols = []
        self.ts_cat_cols = []
        if ts_categorical_cols is not None:
            self.ts_cat_cols.extend(ts_categorical_cols)
        if ts_numeric_cols is not None:
            self.ts_num_cols.extend(ts_numeric_cols)
        self.ts_dat_cols = []
        self.static_transformers: Dict[str, ColumnTransformer] = {}
        self.trend_transformers: Dict[str, TrendColumnTransformer] = {}
        self.len_transformer = column_transformers[DataType.numeric](**self._column_params[DataType.numeric]) \
            if agg_length else None
        self.max_len = 0

    def fit(self, data: TSData):
        """
        Fit the data transformer.

        Parameters
        ----------
        data : TSData
            The data to fit transformer on.
        """
        start_time = time.time()
        if data.static_data is not None:
            static_data = data.static_data.T
            for c in tqdm(static_data.columns, "Fitting static columns"):
                col_value = static_data[c]
                if c in self.st_num_cols:
                    dtype = DataType.numeric
                elif c in self.st_cat_cols:
                    dtype = DataType.categorical
                else:
                    if col_value.nunique() < self.min_n_continuous:
                        self.st_cat_cols.append(c)
                        dtype = DataType.categorical
                    elif pd.to_numeric(col_value, errors="coerce").notna().all():
                        self.st_num_cols.append(c)
                        dtype = DataType.numeric
                    elif pd.to_datetime(
                            col_value, errors="coerce", dayfirst=self.dayfirst, yearfirst=self.yearfirst
                    ).notna().all():
                        self.st_dat_cols.append(c)
                        dtype = DataType.datetime
                    else:
                        self.st_cat_cols.append(c)
                        dtype = DataType.categorical
                transformer = column_transformers[dtype](**self._column_params[dtype])
                transformer.fit(static_data[c])
                self.static_transformers[c] = transformer

        num_possible = collections.defaultdict(lambda: True)
        dat_possible = collections.defaultdict(lambda: True)
        columns = None
        lengths = {}
        for sid, ts_data, _ in data:
            if columns is None:
                columns = ts_data.columns
            else:
                if not columns.equals(ts_data.columns):
                    raise ValueError("Columns from different timeseries do not match.")
            lengths[sid] = ts_data.shape[0]
            for c in columns:
                col_value = ts_data[c]
                if pd.to_numeric(col_value, errors="coerce").isna().any():
                    num_possible[c] = False
                if pd.to_datetime(
                        col_value, errors="coerce", dayfirst=self.dayfirst, yearfirst=self.yearfirst
                ).isna().any():
                    dat_possible[c] = False
        lengths = pd.Series(lengths)
        if self.len_transformer is not None:
            self.len_transformer.fit(lengths)
        self.max_len = lengths.max()

        for c in tqdm(columns, "Fitting time series columns"):
            col_data = {sid: ts_data[c] for sid, ts_data, _ in data}
            nunique = pd.concat(col_data).nunique()
            if c in self.ts_num_cols:
                dtype = DataType.numeric
            elif c in self.ts_cat_cols:
                dtype = DataType.categorical
            else:
                if nunique < self.min_n_continuous:
                    self.ts_cat_cols.append(c)
                    dtype = DataType.categorical
                elif num_possible[c]:
                    self.ts_num_cols.append(c)
                    dtype = DataType.numeric
                elif dat_possible[c]:
                    self.ts_dat_cols.append(c)
                    dtype = DataType.datetime
                else:
                    self.ts_cat_cols.append(c)
                    dtype = DataType.categorical
            transformer = trend_transformers[dtype](**self._trend_params[dtype])
            transformer.fit(col_data)
            self.trend_transformers[c] = transformer
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Fitting transformer time: {total_time_str}")

    def get_static(self, data: TSData, normalize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get static data, including static columns and aggregated timeseries information.

        Parameters
        ----------
        data : TSData
            The raw timeseries data.
        normalize : bool
            Whether to return normalized result or standardized result only.

        Returns
        -------
        pd.DataFrame
            The obtained static data.
        pd.DataFrame
            The aggregated data (can also be regarded as static).
        """
        start_time = time.time()
        if data.static_data is not None:
            static_data = data.static_data.T
            static_data = self.transform_static(static_data, normalize)
        else:
            static_data = pd.DataFrame(index=data.static_ids)

        aggregated = {}
        if self.len_transformer is not None:
            lengths = {}
            for sid, ts_data, _ in data:
                lengths[sid] = ts_data.shape[0]
            lengths = pd.Series(lengths)
            if normalize:
                lengths = self.len_transformer.normalize(lengths)
            else:
                lengths = self.len_transformer.standardize(lengths)
            aggregated[".length"] = lengths
        for c, transformer in tqdm(self.trend_transformers.items(), "Aggregating"):
            col_data = {sid: ts_data[c] for sid, ts_data, _ in data}
            aggregated[c] = transformer.aggregate(col_data, normalize=normalize)
        aggregated = pd.concat(aggregated, axis=1)
        aggregated = flatten_columns(aggregated)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Getting static (normalized={normalize}) time: {total_time_str}")
        return static_data, aggregated

    def transform_static(self, static: pd.DataFrame, normalize: bool) -> pd.DataFrame:
        """
        Transform static data.

        Parameters
        ----------
        static : pd.DataFrame
            The static data in raw format.
        normalize : bool
            Whether to normalize the result (otherwise standardize only).

        Returns
        -------
        pd.DataFrame
            Transformed static data.
        """
        if len(self.static_transformers) == 0:
            return pd.DataFrame(index=static.index)
        result = {}
        for c, transformer in tqdm(self.static_transformers.items(), "Transforming static"):
            if normalize:
                result[c] = transformer.normalize(static[c])
            else:
                result[c] = transformer.standardize(static[c])
        static = pd.concat(result, axis=1)
        static = flatten_columns(static)
        return static

    def get_timeseries(self, data: TSData, chunk_size: int = 100, cache_dir: str = "data-cache") -> TSData:
        """
        Get timeseries data, including static components (normalized version of `.get_static` result).

        Parameters
        ----------
        data : TSData
            The raw timeseries data.
        chunk_size : int
            The number of timeseries to process simultaneously. Timeseries are not handled one by one for efficiency
            concern, but not all at once in case of excessively large dataset that results in OOM.
        cache_dir : str
            The data cache directory for the transformed timeseries data.

        Returns
        -------
        TSData
            The transformed timeseries data for model training.
        """
        all_static = []
        all_aggregated = []
        lengths = {}
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "ts"), exist_ok=True)
        pbar = tqdm(desc="Transforming timeseries", total=len(data) * len(self.trend_transformers))
        columns = None
        for st in range(0, len(data), chunk_size):
            ts_data, static_data = data.get_batch(slice(st, st + chunk_size))
            batch_size = static_data.shape[0]

            all_static.append(static_data)

            aggregated = {}
            ts_transformed = collections.defaultdict(dict)
            for c, transformer in self.trend_transformers.items():
                col_data = {sid: d[c] for sid, d in ts_data.items()}
                aggregated[c] = transformer.aggregate(col_data, normalize=True)
                for sid, d in transformer.transform(col_data).items():
                    ts_transformed[sid][c] = d
                pbar.update(batch_size)
            aggregated = pd.concat(aggregated, axis=1)
            if columns is None:
                columns = aggregated.columns
            else:
                if not columns.equals(aggregated.columns):
                    raise ValueError("Columns from different timeseries do not match.")
            aggregated = flatten_columns(aggregated)
            all_aggregated.append(aggregated.loc[static_data.index])


            for sid, one_ts_data in ts_transformed.items():
                one_ts_data = pd.concat(one_ts_data, axis=1)
                one_ts_data = flatten_columns(one_ts_data)
                one_ts_data.to_csv(os.path.join(cache_dir, "ts", f"{sid}.csv"), index=False)
                if self.len_transformer is not None:
                    lengths[sid] = one_ts_data.shape[0]

        all_static = pd.concat(all_static)
        if all_static.shape[-1] > 0:
            static_transformed = {}
            for c, transformer in self.static_transformers.items():
                static_transformed[c] = transformer.normalize(all_static[c])
            static_transformed = pd.concat(static_transformed, axis=1)
            static_transformed = flatten_columns(static_transformed)
        else:
            static_transformed = all_static

        # columns = all_aggregated[0].columns
        # for i, item in enumerate(all_aggregated):
        #     if not item.columns.equals(columns):
        #         logger.warning(f"Columns mismatch in aggregated data at index {i}: {item.columns.tolist()} vs {columns.tolist()}")
        #         missing_cols = columns.difference(item.columns)
        #         if len(missing_cols) > 0:
        #             logger.warning(f"Missing columns in aggregated data at index {i}: {missing_cols.tolist()}")
                
        all_aggregated = pd.concat(all_aggregated)
        
        if self.len_transformer is not None:
            lengths = pd.Series(lengths)
            lengths = self.len_transformer.normalize(lengths)
            lengths.columns = [f".length.{c}" for c in lengths.columns]
        else:
            lengths = pd.DataFrame(index=all_aggregated.index)

        static_combined = pd.concat([static_transformed, lengths, all_aggregated], axis=1).loc[static_transformed.index]

        static_combined.to_csv(os.path.join(cache_dir, "static.csv"), index_label=".id")

        all_ts_data = TSData(os.path.join(cache_dir, "ts"), os.path.join(cache_dir, "static.csv"), ".id")
        return all_ts_data

    def get_norm_ts_cond(self, static: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
        """
        From standardized static and aggregated data (output of `.get_static`), create the conditions for timeseries
        model (normalized).

        Parameters
        ----------
        static : pd.DataFrame
            The static data.
        agg : pd.DataFrame
            The aggregated data.

        Returns
        -------
        pd.DataFrame
            Combined timeseries conditions.
        """
        static = self.recover_static_standardized(static)
        static = self.transform_static(static, normalize=True)

        if agg is not None:
            if self.len_transformer is not None:
                len_data = extract(agg, ".length")
                recov_len = self.len_transformer.inverse_standardize(len_data)
                length = self.len_transformer.normalize(recov_len)
                length = length.set_axis([f".length.{c}" for c in length.columns], axis=1)
            else:
                length = pd.DataFrame(index=agg.index)

            agg_result = {}
            for c, transformer in self.trend_transformers.items():
                agg_data = extract(agg, c)
                agg_data = transformer.normalize_aggregated(agg_data)
                agg_result[c] = agg_data
            agg_result = pd.concat(agg_result, axis=1)
            agg_result = flatten_columns(agg_result)
            return pd.concat([static, length, agg_result], axis=1)
        else:
            return static

    def recover(
            self, data: TSData, static_standardized: Optional[pd.DataFrame] = None,
            chunk_size: int = 100, cache_dir: str = "data-cache"
    ) -> TSData:
        """
        Recover normalized timeseries data to raw format. This is the inverse process of `.get_timeseries` (or
        `.get_static` too).

        Parameters
        ----------
        data : TSData
            The transformed timeseries data, which is typically obtained by `get_timeseries`.
        static_standardized : pd.DataFrame, optional
            The standardized static data, by `get_static` with `normalize=False`. If this is not provided, then we
            will use the normalized static data from `data`.
        chunk_size : int
            Chunk size similarly to `.get_timeseries`.
        cache_dir : str
            The output cache directory for the recovered timeseries data.

        Returns
        -------
        TSData
            The recovered timeseries data.
        """
        if len(self.static_transformers) == 0:
            static = None
        elif static_standardized is not None:
            static = self.recover_static_standardized(static_standardized)
        else:
            result = {}
            for c, transformer in self.static_transformers.items():
                col_normalized = extract(data.static_data.T, c)
                result[c] = transformer.inverse_normalize(col_normalized)
            static = pd.DataFrame(result)

        os.makedirs(cache_dir, exist_ok=True)
        if static is not None:
            static.to_csv(os.path.join(cache_dir, "static.csv"), index_label=".id")
        os.makedirs(os.path.join(cache_dir, "ts"), exist_ok=True)
        for st in range(0, len(data), chunk_size):
            ts_data, static_data = data.get_batch(slice(st, st + chunk_size))
            ts_recovered = collections.defaultdict(dict)
            for c, transformer in self.trend_transformers.items():
                col_data = {sid: extract(d, c) for sid, d in ts_data.items()}
                col_recovered = transformer.inverse_transform(col_data, extract(static_data, c), True)
                for sid, d in col_recovered.items():
                    ts_recovered[sid][c] = d
            for sid, one_ts_data in ts_recovered.items():
                pd.DataFrame(one_ts_data).to_csv(os.path.join(cache_dir, "ts", f"{sid}.csv"), index=False)

        return TSData(os.path.join(cache_dir, "ts"), os.path.join(cache_dir, "static.csv"), ".id")

    def recover_static_standardized(self, static_standardized: pd.DataFrame) -> pd.DataFrame:
        """
        Recover static standardized data.

        Parameters
        ----------
        static_standardized : pd.DataFrame
            The standardized static data.

        Returns
        -------
        pd.DataFrame
            Recovered static data in raw format.
        """
        if len(self.static_transformers) == 0:
            return pd.DataFrame(index=static_standardized.index)
        result = {}
        for c, transformer in self.static_transformers.items():
            col_standardized = extract(static_standardized, c)
            result[c] = transformer.inverse_standardize(col_standardized)
        static = pd.DataFrame(result)
        return static

    @property
    def static_standardized_types(self) -> Tuple[Dict[str, DataType], Dict[str, DataType]]:
        """
        Static standardized data types, with static and aggregated parts separated.
        """
        static_types = {
            f"{c}.{sc}": t for c, transformer in self.static_transformers.items()
            for sc, t in transformer.standardized_types.items()
        }
        len_types = {} if self.len_transformer is None else {
            f".length.{sc}": t for sc, t in self.len_transformer.standardized_types.items()
        }
        agg_types = {
            f"{c}.{sc}": t for c, transformer in self.trend_transformers.items()
            for sc, t in transformer.standardized_aggregated_types.items()
        }
        return static_types, len_types | agg_types

    @property
    def static_spans(self) -> Tuple[List[Tuple[int, SpanType]], List[Tuple[int, SpanType]]]:
        """
        Static data spans, with static and aggregated parts separated.
        """
        results = []
        for c, transformer in self.static_transformers.items():
            results.extend(transformer.spans)
        agg_results = []
        if self.len_transformer is not None:
            agg_results.extend(self.len_transformer.spans)
        for c, transformer in self.trend_transformers.items():
            agg_results.extend(transformer.aggregated_spans)
        results = [(w, t) for w, t in results if w > 0]
        agg_results = [(w, t) for w, t in agg_results if w > 0]
        return results, agg_results

    @property
    def transformed_columns(self) -> List[str]:
        """
        Timeseries transformed columns.
        """
        results = []
        for c, transformer in self.trend_transformers.items():
            for sc in transformer.transformed_columns:
                results.append(f"{c}.{sc}")
        return results

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        """
        Timeseries data spans.
        """
        results = []
        for c, transformer in self.trend_transformers.items():
            results.extend(transformer.spans)
        results = [(w, t) for w, t in results if w > 0]
        return results
