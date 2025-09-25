import inspect
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core import groupby as pdg

from .column import ColumnTransformer, DataType, NumericTransformer, SpanType, column_transformers, dt_components
from .utils import extract, flatten_columns


class TrendColumnTransformer(ABC):
    """
    Data transformer for a column with trend (in timeseries).
    """
    dtype: DataType = None
    """
    The data type for this transformer.
    """
    def __init__(self, n_first: int = 1, n_last: int = 0, **kwargs):
        """
        Parameters
        ----------
        n_first: int
            The number of first steps to be maintained in aggregated information.
        n_last: int
            The number of last steps to be maintained in aggregated information.
        **kwargs
            Other arguments to the corresponding column's plain transformer.
        """
        self.n_first = n_first
        self.n_last = n_last
        self.column_transformer = column_transformers[self.dtype](**kwargs)

    params = set(inspect.signature(__init__).parameters) - {"self", "kwargs"}

    def fit(self, data: Dict[str, pd.Series]):
        """
        Fit this transformer.

        Parameters
        ----------
        data : Dict[str, pd.Series]
            The column's data of different timeseries. Each value in the dict is a timeseries (of this column),
            corresponding to the timeseries with static ID in the key.
        """
        self.column_transformer.fit(pd.concat(data.values(), ignore_index=True))
        self._fit(data)

    @abstractmethod
    def _fit(self, data: Dict[str, pd.Series]):
        raise NotImplementedError()

    def aggregate(self, data: Dict[str, pd.Series], normalize: bool = False) -> pd.DataFrame:
        """
        Aggregate the timeseries into vectorized presentations.

        Parameters
        ----------
        data : Dict[str, pd.Series]
            The column's data of different timeseries.
        normalize : bool
            Whether to normalize the result. If not, the value will be standardized.

        Returns
        -------
        pd.DataFrame
            The aggregated result. The index of the dataframe is the static IDs.
        """
        grouped = pd.concat({k: v.reset_index(drop=True) for k, v in data.items()}).groupby(level=0)
        standardized = self.column_transformer.standardize(grouped.obj)
        g_standardized = standardized.groupby(axis=0, level=0)
        first = g_standardized.head(self.n_first).unstack(level=1)
        if self.n_last > 0:
            tail = g_standardized.tail(self.n_last)
            tail = pd.concat({k: v.reset_index(drop=True) for k, v in tail.groupby(axis=0, level=0)})
            last = tail.unstack(level=1)
        else:
            last = g_standardized.tail(self.n_last).unstack(level=1)
        if normalize:
            first = first.swaplevel(0, 1, axis=1)
            last = last.swaplevel(0, 1, axis=1)
            if self.n_first > 0:
                normalized_first = {
                    i: self._normalize_standardized(first[i]) for i in range(self.n_first)
                }
                first = pd.concat(normalized_first, axis=1).swaplevel(0, 1, axis=1)
            if self.n_last > 0:
                normalized_last = {
                    i: self._normalize_standardized(last[i]) for i in range(self.n_last)
                }
                last = pd.concat(normalized_last, axis=1).swaplevel(0, 1, axis=1)
        first.columns = [f"first-{i}.{c}" for c, i in first.columns]
        last.columns = [f"last-{i}.{c}" for c, i in last.columns]
        custom = self._aggregate(grouped, normalize)
        return pd.concat([custom, first, last], axis=1)

    @abstractmethod
    def _aggregate(self, data: pdg.SeriesGroupBy, normalize: bool = False) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def _normalize_standardized(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @property
    def standardized_aggregated_types(self) -> Dict[str, DataType]:
        """
        The standardized column types of aggregated result. The result will contain categorical or numeric type.
        """
        return self._standardized_aggregated_types() | {
            f"first-{i}.{c}": t for c, t in self.column_transformer.standardized_types.items()
            for i in range(self.n_first)
        } | {
            f"last-{i}.{c}": t for c, t in self.column_transformer.standardized_types.items()
            for i in range(self.n_last)
        }

    @abstractmethod
    def _standardized_aggregated_types(self) -> Dict[str, DataType]:
        raise NotImplementedError()

    @property
    def aggregated_spans(self) -> List[Tuple[int, SpanType]]:
        """
        Spans of aggregated normalized result.
        """
        return self._aggregated_spans() + self.column_transformer.spans * (self.n_first + self.n_last)

    @abstractmethod
    def _aggregated_spans(self) -> List[Tuple[int, SpanType]]:
        raise NotImplementedError()

    def normalize_aggregated(self, standardized_aggregated: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the standardized aggregated result.

        Parameters
        ----------
        standardized_aggregated : pd.DataFrame
            The standardized aggregated data.

        Returns
        -------
        pd.DataFrame
            The normalized result.
        """
        results = {}
        for i in range(self.n_first):
            step_col = extract(standardized_aggregated, f"first-{i}")
            step_col = self._normalize_standardized(step_col)
            results[f"first-{i}"] = step_col
        for i in range(self.n_last):
            step_col = extract(standardized_aggregated, f"last-{i}")
            step_col = self._normalize_standardized(step_col)
            results[f"last-{i}"] = step_col
        if len(results) > 0:
            first_last = pd.concat(results, axis=1)
            first_last = flatten_columns(first_last)
        else:
            first_last = pd.DataFrame(index=standardized_aggregated.index)
        custom = self._normalize_aggregated(standardized_aggregated[[
            c for c in standardized_aggregated.columns if
            not re.fullmatch(r"first-\d+\..+", c) and not re.fullmatch("last-\d+\..+", c)
        ]])
        return pd.concat([custom, first_last], axis=1)

    @abstractmethod
    def _normalize_aggregated(self, standardized_aggregated: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        """
        Transform the timeseries into neural network-friendly numeric presentations.

        Parameters
        ----------
        data : Dict[str, pd.Series]
            The column's data of different timeseries.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The transformed timeseries.
        """
        raise NotImplementedError()

    @abstractmethod
    def inverse_transform(
            self, ts_data: Dict[str, pd.DataFrame], aggregated: pd.DataFrame, normalized_aggregated: bool = False
    ) -> Dict[str, pd.Series]:
        """
        Recover raw data from transformed data.

        Parameters
        ----------
        ts_data : Dict[str, pd.DataFrame]
            The result of `transform`.
        aggregated : pd.DataFrame
            The result of `aggregate`.
        normalized_aggregated : bool
            The `normalized` argument to `aggregate` when creating `aggregated`.

        Returns
        -------
        Dict[str, pd.Series]
            Recovered raw timeseries data.
        """
        raise ValueError()

    @property
    @abstractmethod
    def transformed_columns(self) -> List[str]:
        """
        Name of transformed columns.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def spans(self) -> List[Tuple[int, SpanType]]:
        """
        Spans of transformed result.
        """
        raise NotImplementedError()


class CategoricalTrendTransformer(TrendColumnTransformer):
    """
    Data transformer for a categorical column with trend (in timeseries).
    """
    dtype = DataType.categorical

    def __init__(self, n_top: int = 1, n_top_prop: int = 10, **kwargs):
        """
        Parameters
        ----------
        n_top: int
            Number of top categories to keep in aggregated information.
        n_top_prop: int
            Number of categories to show proportions in the timeseries, more frequently used categories are prioritized.
            If negative values are provided, all categories are shown.
        **kwargs
            Other arguments to parent class.
        """
        self.n_top = n_top
        self.n_top_prop = n_top_prop
        self._prop_columns = []
        super().__init__(**kwargs)

    params = (set(inspect.signature(column_transformers[dtype]).parameters) |
              set(inspect.signature(TrendColumnTransformer).parameters) |
              set(inspect.signature(__init__).parameters)) - {"self", "kwargs"}

    def _fit(self, data: Dict[str, pd.Series]):
        if self.n_top_prop < 0:
            self._prop_columns = self.column_transformer.oe.categories_[0].tolist()
        else:
            grouped = pd.concat({k: v.reset_index(drop=True) for k, v in data.items()}).groupby(level=0)
            cnts = grouped.value_counts().unstack(level=1).fillna(0).sum()
            n_top_prop = min(self.n_top_prop, cnts.shape[0])
            self._prop_columns = cnts.nlargest(n_top_prop).index.tolist()
        self.n_top = min(self.n_top, len(self.column_transformer.oe.categories_[0]))

    def _aggregate(self, data: pdg.SeriesGroupBy, normalize: bool = False) -> pd.DataFrame:
        cnts = data.value_counts(normalize=True).unstack(level=1).fillna(0)
        if self.n_top > 0:
            top_values = []
            vcnts = cnts.copy().values
            for i in range(self.n_top):
                argmax = np.argmax(vcnts, axis=1)
                top_values.append(pd.Series(cnts.columns[argmax], index=cnts.index, dtype=data.obj.dtype))
                vcnts[np.arange(vcnts.shape[0]), argmax] = -1
            transformed_top_values = []
            for i, v in enumerate(top_values):
                transformed = self.column_transformer.normalize(v) if normalize \
                    else self.column_transformer.standardize(v)
                transformed.columns = [f"top-{i}.{c}" for c in transformed.columns]
                transformed_top_values.append(transformed)
            tops = pd.concat(transformed_top_values, axis=1)
        else:
            tops = cnts[[]]
            
        # Step 1: Convert all column names in cnts to strings
        cnts.columns = cnts.columns.astype(str)

        # Step 2: Make sure self._prop_columns are also strings
        prop_cols_str = [str(c) for c in self._prop_columns]

        # Step 3: Add any missing columns (with string keys)
        for col in prop_cols_str:
            if col not in cnts.columns:
                cnts[col] = 0.0

        # Step 4: Reorder and rename
        cnts = cnts[prop_cols_str]
        cnts.columns = [f"prop-{c}" for c in prop_cols_str]


        return pd.concat([tops, cnts], axis=1)

    def _normalize_standardized(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.column_transformer.normalize(data["val"])

    def _standardized_aggregated_types(self) -> Dict[str, DataType]:
        return {
            f"top-{i}.{c}": t for c, t in self.column_transformer.standardized_types.items() for i in range(self.n_top)
        } | {f"prop-{c}": DataType.numeric for c in self._prop_columns}

    def _aggregated_spans(self) -> List[Tuple[int, SpanType]]:
        return self.column_transformer.spans * self.n_top + [(len(self._prop_columns), SpanType.continuous)]

    def _normalize_aggregated(self, standardized_aggregated: pd.DataFrame) -> pd.DataFrame:
        if self.n_top > 0:
            tops = {}
            for i in range(self.n_top):
                top_val = extract(standardized_aggregated, f"top-{i}")
                top_val = self._normalize_standardized(top_val)
                tops[f"top-{i}"] = top_val
            tops = pd.concat(tops, axis=1)
            tops = flatten_columns(tops)
        else:
            tops = pd.DataFrame(index=standardized_aggregated.index)
        prop_data = standardized_aggregated[[f"prop-{c}" for c in self._prop_columns]]
        return pd.concat([tops, prop_data], axis=1)

    def transform(self, data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        grouped = pd.concat(data).groupby(level=0)
        transformed = self.column_transformer.normalize(grouped.obj)
        g_transformed = transformed.groupby(axis=0, level=0)
        return {k: v.droplevel(0) for k, v in g_transformed}

    def inverse_transform(
            self, ts_data: Dict[str, pd.DataFrame], aggregated: pd.DataFrame, normalized_aggregated: bool = False
    ) -> Dict[str, pd.Series]:
        grouped = pd.concat(ts_data)
        recovered = self.column_transformer.inverse_normalize(grouped)
        return {k: v.droplevel(0) for k, v in recovered.groupby(level=0)}

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        return self.column_transformer.spans

    @property
    def transformed_columns(self) -> List[str]:
        return self.column_transformer.normalized_columns


class NumericTrendTransformer(TrendColumnTransformer):
    """
    Data transformer for a numeric column with trend (in timeseries).
    """
    dtype = DataType.numeric

    def __init__(
            self, max_order: int = 1, max_agg_order: int = 0,
            agg_functions: Literal["mean", "std", "min", "max", "var", "median", "sum"] = ["mean", "std"], **kwargs
    ):
        """
        Parameters
        ----------
        max_order : int
            The maximum order of differentiation for the transformed timeseries data.
            Order-0 differentiation means the original data. Order-1 differentiation means the difference of current
            step and previous step. Higher order differentiation takes the difference of the differentiated value of
            one order lower of the current step and previous step.
        max_agg_order : int
            Maximum order of differentiation to aggregate.
        agg_functions : "mean" | "std" | "min" | "max" | "var" | "median" | "sum"
            Aggregation functions.
        **kwargs
            Other arguments to parent class.
        """
        self.max_order = max_order
        self.max_agg_order = max_agg_order
        if max_agg_order > max_order:
            raise ValueError("Maximum aggregation order must not be larger than maximum order.")
        self.agg_functions = agg_functions
        params = {
            k: v for k, v in kwargs.items() if k in NumericTransformer.params
        }
        self._higher_order_transformers = {
            i: NumericTransformer(**params) for i in range(1, self.max_order + 1)
        }
        minimum_params = {
            k: v if k != "n_bins" else -1 for k, v in params.items()
        }
        if "n_bins" not in params:
            minimum_params["n_bins"] = -1
        self._agg_transformers: Dict[Tuple[int, str], NumericTransformer] = {
            (i, f): NumericTransformer(**minimum_params)
            for i in range(self.max_agg_order + 1) for f in self.agg_functions
        }
        super().__init__(**kwargs)

    params = (set(inspect.signature(column_transformers[dtype]).parameters) |
              set(inspect.signature(TrendColumnTransformer).parameters) |
              set(inspect.signature(__init__).parameters)) - {"self", "kwargs"}

    def _fit(self, data: Dict[str, pd.Series]):
        grouped = pd.concat({k: v.reset_index(drop=True) for k, v in data.items()}).groupby(level=0)
        aggregated = grouped.aggregate(self.agg_functions).fillna(0)
        for c in self.agg_functions:
            self._agg_transformers[0, c].fit(aggregated[c])
        for i in range(1, self.max_order + 1):
            grouped = grouped.diff().bfill()
            self._higher_order_transformers[i].fit(grouped)
            grouped = grouped.groupby(level=0)
            if i <= self.max_agg_order:
                aggregated = grouped.aggregate(self.agg_functions).fillna(0)
                for c in self.agg_functions:
                    self._agg_transformers[i, c].fit(aggregated[c])
        self._higher_order_transformers[0] = self.column_transformer

    def _aggregate(self, data: pdg.SeriesGroupBy, normalize: bool = False) -> pd.DataFrame:
        aggregated_by_order = []
        for i in range(self.max_agg_order + 1):
            if len(self.agg_functions) > 0:
                aggregated = data.aggregate(self.agg_functions).fillna(0)
                agg_transformed = {}
                for c in self.agg_functions:
                    transformer = self._agg_transformers[i, c]
                    agg_col = aggregated[c]
                    transformed = transformer.normalize(agg_col) if normalize else transformer.standardize(agg_col)
                    agg_transformed[c] = transformed
                agg_transformed = pd.concat(agg_transformed, axis=1)
                agg_transformed.columns = [f"{a}{i}.{c}" for a, c in agg_transformed.columns]
            else:
                agg_transformed = pd.DataFrame(index=data.grouper.group_keys_seq)
            aggregated_by_order.append(agg_transformed)
            if i < self.max_agg_order:
                data = data.diff().bfill().groupby(level=0)
        aggregated_by_order = reversed(aggregated_by_order)
        return pd.concat(aggregated_by_order, axis=1)

    def _normalize_standardized(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.column_transformer.normalize(data["val"])

    def _standardized_aggregated_types(self) -> Dict[str, DataType]:
        return {
            f"{a}{i}.{c}": t for (i, a), transformer in self._agg_transformers.items()
            for c, t in transformer.standardized_types.items()
        }

    def _aggregated_spans(self) -> List[Tuple[int, SpanType]]:
        results = []
        for i in reversed(range(self.max_agg_order + 1)):
            width = 0
            for a in self.agg_functions:
                spans = self._agg_transformers[i, a].spans
                if any(t == SpanType.discrete for _, t in spans):
                    raise ValueError(f"Aggregated numeric value ({a}) should not have discrete spans.")
                if len(spans) > 1:
                    raise ValueError(f"Aggregated numeric value ({a}) should have singleton normalizer.")
                width += spans[0][0]
            results.append((width, SpanType.continuous))
        return results

    def _normalize_aggregated(self, standardized_aggregated: pd.DataFrame) -> pd.DataFrame:
        if len(self.agg_functions) > 0:
            result = {}
            for i in range(self.max_agg_order + 1):
                for c in self.agg_functions:
                    transformer = self._agg_transformers[i, c]
                    agg_val = extract(standardized_aggregated, f"{c}{i}")
                    agg_val = transformer.normalize(agg_val["val"])
                    result[f"{c}{i}"] = agg_val
            result = pd.concat(result, axis=1)
            return flatten_columns(result)
        else:
            return standardized_aggregated[[]]

    def transform(self, data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        grouped = pd.concat(data).groupby(level=0)
        transformed_by_order = []
        for i in range(self.max_order + 1):
            transformed = self._higher_order_transformers[i].normalize(grouped.obj)
            transformed_by_order.append(transformed)
            if i < self.max_order:
                grouped = grouped.diff().bfill().groupby(level=0)
        transformed_by_order = reversed(transformed_by_order)
        all_transformed = pd.concat(
            {f"order{self.max_order - i}": transformed for i, transformed in enumerate(transformed_by_order)}, axis=1
        )
        all_transformed = flatten_columns(all_transformed)
        return {k: v.droplevel(0) for k, v in all_transformed.groupby(axis=0, level=0)}

    def inverse_transform(
            self, ts_data: Dict[str, pd.DataFrame], aggregated: pd.DataFrame, normalized_aggregated: bool = False
    ) -> Dict[str, pd.Series]:
        combined = pd.concat(ts_data)
        order0 = extract(combined, "order0")
        recovered = self.column_transformer.inverse_normalize(order0)
        return {k: v.droplevel(0) for k, v in recovered.groupby(level=0)}

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        results = []
        for i in reversed(range(self.max_order + 1)):
            results.extend(self._higher_order_transformers[i].spans)
        return results

    @property
    def transformed_columns(self) -> List[str]:
        results = []
        for i in reversed(range(self.max_order + 1)):
            for c in self._higher_order_transformers[i].normalized_columns:
                results.append(f"order{i}.{c}")
        return results


class DatetimeTrendTransformer(TrendColumnTransformer):
    """
    Data transformer for a datetime column with trend (in timeseries).
    """
    dtype = DataType.datetime

    def __init__(
            self,
            trend_components: List[dt_components] = ["day_name", "month_name"],
            **kwargs
    ):
        """
        Parameters
        ----------
        trend_components : List[dt_components]
            The components to maintain a non-trivial trend transformer on.
        **kwargs
            Other arguments for parent class and categorical or numeric trend transformers.
        """
        super().__init__(**{
            k: v for k, v in kwargs.items() if k in TrendColumnTransformer.params or
            k in column_transformers[self.dtype].params
        })
        self.trend_components = trend_components
        self.auxiliary_trend_transformers: Dict[dt_components, TrendColumnTransformer] = {}
        self.core_trend_transformer = NumericTrendTransformer(**{
            k: v for k, v in kwargs.items() if k in NumericTrendTransformer.params
        })
        self._trend_params = {
            t: {
                k: v for k, v in kwargs.items() if k in transformer.params
            } for t, transformer in trend_transformers.items()
        }

    params = (set(inspect.signature(column_transformers[dtype]).parameters) |
              set(inspect.signature(TrendColumnTransformer).parameters) |
              set(inspect.signature(CategoricalTrendTransformer).parameters) |
              set(inspect.signature(NumericTrendTransformer).parameters) |
              set(inspect.signature(__init__).parameters)) - {"self", "kwargs"}

    def _fit(self, data: Dict[str, pd.Series]):
        combined = pd.concat({k: v.reset_index(drop=True) for k, v in data.items()})
        standardized = self.column_transformer.standardize(combined)
        g_standardized = standardized.groupby(axis=0, level=0)
        for component in self.trend_components:
            component_transformer = self.column_transformer.components[component]
            if isinstance(component_transformer, ColumnTransformer):
                transformer = trend_transformers[component_transformer.dtype](
                    **self._trend_params[component_transformer.dtype]
                )
                transformer.fit({g: d for g, d in g_standardized[f"{component}.val"]})
                self.auxiliary_trend_transformers[component] = transformer
        self.core_trend_transformer.fit({g: d for g, d in g_standardized["val.val"]})

    def aggregate(self, data: Dict[str, pd.Series], normalize: bool = False) -> pd.DataFrame:
        data = pd.concat(data).groupby(level=0)
        standardized = self.column_transformer.standardize(data.obj)
        g_standardized = standardized.groupby(axis=0, level=0)
        aggregated = {}
        for component, transformer in self.auxiliary_trend_transformers.items():
            aggregated[component] = transformer.aggregate(
                {g: d for g, d in g_standardized[f"{component}.val"]}, normalize
            )
        aggregated["val"] = self.core_trend_transformer.aggregate(
            {g: d for g, d in g_standardized["val.val"]}, normalize
        )
        aggregated = pd.concat(aggregated, axis=1)
        aggregated = flatten_columns(aggregated)
        return aggregated

    def _aggregate(self, data: pdg.SeriesGroupBy, normalize: bool = False) -> pd.DataFrame:
        return pd.DataFrame(index=data.grouper.group_keys_seq)

    def _normalize_standardized(self, data: pd.DataFrame) -> pd.DataFrame:
        normalized = {}
        for component, transformer in self.column_transformer.components.items():
            normalized[component] = transformer.normalize(data[f"{component}.val"])
        normalized["val"] = self.column_transformer.core_transformer.normalize(data["val.val"])
        normalized = pd.concat(normalized, axis=1)
        normalized = flatten_columns(normalized)
        return normalized

    @property
    def standardized_aggregated_types(self) -> Dict[str, DataType]:
        results = {}
        for component, transformer in self.auxiliary_trend_transformers.items():
            for c, t in transformer.standardized_aggregated_types.items():
                results[f"{component}.{c}"] = t
        for c, t in self.core_trend_transformer.standardized_aggregated_types.items():
            results[f"val.{c}"] = t
        return results

    def _standardized_aggregated_types(self) -> Dict[str, DataType]:
        return {}

    @property
    def aggregated_spans(self) -> List[Tuple[int, SpanType]]:
        results = []
        for component, transformer in self.auxiliary_trend_transformers.items():
            results.extend(transformer.aggregated_spans)
        results.extend(self.core_trend_transformer.aggregated_spans)
        return results

    def _aggregated_spans(self) -> List[Tuple[int, SpanType]]:
        return []

    def normalize_aggregated(self, standardized_aggregated: pd.DataFrame) -> pd.DataFrame:
        results = {}
        for component, transformer in self.auxiliary_trend_transformers.items():
            component_data = extract(standardized_aggregated, component)
            results[component] = transformer.normalize_aggregated(component_data)
        core_data = extract(standardized_aggregated, "val")
        results["val"] = self.core_trend_transformer.normalize_aggregated(core_data)
        results = pd.concat(results, axis=1)
        return flatten_columns(results)

    def _normalize_aggregated(self, standardized_aggregated: pd.DataFrame) -> pd.DataFrame:
        return standardized_aggregated[[]]

    def transform(self, data: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        combined = pd.concat(data)
        standardized = self.column_transformer.standardize(combined)
        g_standardized = standardized.groupby(axis=0, level=0)
        normalized = self.column_transformer.normalize(combined)
        results = {}
        for component in self.column_transformer.auxiliary_components:
            component_transformer = self.column_transformer.components[component]
            if not isinstance(component_transformer, ColumnTransformer):
                continue
            normalized_component = extract(normalized, component)
            if component in self.auxiliary_trend_transformers:
                trend_transformed = pd.concat(self.auxiliary_trend_transformers[component].transform({
                    g: d.droplevel(0) for g, d in g_standardized[f"{component}.val"]
                }))
                drop_normalized = []
                for c in normalized_component:
                    if c == "val" or re.fullmatch(r"cat-\d\d+", c):
                        drop_normalized.append(c)
                results[component] = pd.concat(
                    [normalized_component.drop(columns=drop_normalized), trend_transformed], axis=1
                )
            else:
                results[component] = normalized_component
        core_transformed = pd.concat(
            self.core_trend_transformer.transform({g: d.droplevel(0) for g, d in g_standardized["val.val"]})
        )
        core_normalized = extract(normalized, "val")
        results["val"] = pd.concat([core_normalized.drop(columns=["val"]), core_transformed], axis=1)
        results = pd.concat(results, axis=1).loc[combined.index]
        results = flatten_columns(results)
        return {g: d.droplevel(0) for g, d in results.groupby(axis=0, level=0)}

    def inverse_transform(
            self, ts_data: Dict[str, pd.DataFrame], aggregated: pd.DataFrame, normalized_aggregated: bool = False
    ) -> Dict[str, pd.Series]:
        recovered = {}
        combined = pd.concat(ts_data)
        for component in self.column_transformer.edit_by_components:
            component_transformer = self.column_transformer.components[component]
            if not isinstance(component_transformer, ColumnTransformer):
                continue
            component_agg, component_ts_normalized, trend_ts_data = self._obtain_component_data(
                combined, aggregated, component, component_transformer
            )
            if component in self.auxiliary_trend_transformers:
                recovered_val = pd.concat(self.auxiliary_trend_transformers[component].inverse_transform(
                    {g: d.droplevel(0) for g, d in trend_ts_data.groupby(axis=0, level=0)},
                    component_agg, normalized_aggregated
                ))
                recovered[component] = recovered_val
            else:
                recovered[component] = self.column_transformer.components[component].inverse_normalize(trend_ts_data)

        core_agg, core_ts_normalized, trend_ts_data = self._obtain_component_data(
            combined, aggregated, "val", self.column_transformer.core_transformer
        )
        recovered_val = pd.concat(self.core_trend_transformer.inverse_transform(
            {g: d.droplevel(0) for g, d in trend_ts_data.groupby(axis=0, level=0)}, core_agg, normalized_aggregated
        ))
        recovered["val"] = recovered_val
        recovered = pd.DataFrame(recovered)
        recovered.columns = [f"{c}.val" for c in recovered.columns]
        recovered = self.column_transformer.inverse_standardize(recovered)
        return {g: d.droplevel(0) for g, d in recovered.groupby(level=0)}

    @staticmethod
    def _obtain_component_data(
            ts_data: pd.DataFrame, aggregated: pd.DataFrame, component: Union[dt_components, Literal["val"]],
            component_transformer: ColumnTransformer
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        component_ts_data = extract(ts_data, component)
        component_agg = extract(aggregated, component)
        component_ts_normalized = component_ts_data[[
            c for c in component_transformer.normalized_columns
            if not (c == "val" or re.fullmatch(r"cat-\d\d+", c))
        ]]
        trend_ts_data = component_ts_data.drop(columns=component_ts_normalized.columns)
        return component_agg, component_ts_normalized, trend_ts_data

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        results = []
        for component in self.column_transformer.auxiliary_components:
            component_transformer = self.column_transformer.components[component]
            if not isinstance(component_transformer, ColumnTransformer):
                continue
            core_spans = component_transformer.spans
            if component in self.auxiliary_trend_transformers:
                trend_spans = self.auxiliary_trend_transformers[component].spans
                spans = core_spans[:-1] + trend_spans
            else:
                spans = core_spans
            results.extend(spans)
        results.extend(self.column_transformer.core_transformer.spans[:-1])
        results.extend(self.core_trend_transformer.spans)
        return results

    @property
    def transformed_columns(self) -> List[str]:
        results = []
        for component in self.column_transformer.auxiliary_components:
            component_transformer = self.column_transformer.components[component]
            if not isinstance(component_transformer, ColumnTransformer):
                continue
            columns = component_transformer.normalized_columns
            if component in self.auxiliary_trend_transformers:
                trend_columns = self.auxiliary_trend_transformers[component].transformed_columns
                last_span_width = component_transformer.spans[-1][0]
                columns = columns[:-last_span_width]
            else:
                trend_columns = []
            for c in columns + trend_columns:
                results.append(f"{component}.{c}")
        core_columns = self.column_transformer.core_transformer.normalized_columns
        last_span_width = self.column_transformer.core_transformer.spans[-1][0]
        core_columns = core_columns[:-last_span_width]
        core_trend_columns = self.core_trend_transformer.transformed_columns
        for c in core_columns + core_trend_columns:
            results.append(f"val.{c}")
        return results


trend_transformers = {
    DataType.categorical: CategoricalTrendTransformer,
    DataType.numeric: NumericTrendTransformer,
    DataType.datetime: DatetimeTrendTransformer,
}