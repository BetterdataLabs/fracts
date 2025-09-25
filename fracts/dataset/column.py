import enum
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from .utils import extract, flatten_columns


class DataType(enum.Enum):
    categorical = enum.auto()
    numeric = enum.auto()
    datetime = enum.auto()


class SpanType(enum.Enum):
    discrete = enum.auto()
    """One-hot discrete spans."""
    continuous = enum.auto()
    """Continuous spans."""


class ColumnTransformer(ABC):
    """
    Data transformer for a column.
    """
    dtype: DataType = None
    """
    The data type for this transformer.
    """
    pd_dtype: str = None
    """
    Pandas data type for intermediate processing the raw data.
    """
    def __init__(self):
        pass

    params = set(inspect.signature(__init__).parameters) - {"self"}

    def fit(self, data: pd.Series):
        """
        Fit this transformer.

        Parameters
        ----------
        data : pd.Series
            The column's data.
        """
        return self._fit(data.astype(self.pd_dtype))

    @abstractmethod
    def _fit(self, data: pd.Series):
        raise NotImplementedError()

    def standardize(self, data: pd.Series) -> pd.DataFrame:
        """
        Standardizes the column data to categorical/numeric data types. This step affects datetime column only.

        Parameters
        ----------
        data : pd.Series
            The column's data.

        Returns
        -------
        pd.DataFrame
            The standardized column data.
        """
        return self._standardize(data.astype(self.pd_dtype))

    @abstractmethod
    def _standardize(self, data: pd.Series) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def inverse_standardize(self, data: pd.DataFrame) -> pd.Series:
        """
        Inverse action of `standardize` step.

        Parameters
        ----------
        data : pd.DataFrame
            The result of `standardize` step.

        Returns
        -------
        pd.Series
            The recovered raw column data.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def standardized_types(self) -> Dict[str, DataType]:
        """
        The standardized column types. The result will contain categorical or numeric type.
        """
        raise NotImplementedError()

    def normalize(self, data: pd.Series) -> pd.DataFrame:
        """
        Normalize the column data to neural network-friendly values. The result will be all numeric values without
        extremely large or small values.

        Parameters
        ----------
        data : pd.Series
            The column's data.

        Returns
        -------
        pd.DataFrame
            The normalized column data.
        """
        return self._normalize(data.astype(self.pd_dtype))

    @abstractmethod
    def _normalize(self, data: pd.Series) -> pd.DataFrame:
        raise NotImplementedError()

    @abstractmethod
    def inverse_normalize(self, data: pd.DataFrame) -> pd.Series:
        """
        Inverse action of `normalize` step.

        Parameters
        ----------
        data : pd.DataFrame
            The result of `normalize` step.

        Returns
        -------
        pd.Series
            The recovered raw column data.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def spans(self) -> List[Tuple[int, SpanType]]:
        """
        Spans of normalized result. The returned value is a list of tuples of (integer, SpanType). Integer stands for
        the width of the span. The sum of the integers in this list is the normalized result's dimensions. The
        normalized result's columns are always in fixed order, and this list of spans corresponding to dimensions from
        left to right.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def normalized_columns(self) -> List[str]:
        """
        Normalized column names.
        """
        raise NotImplementedError()


class CategoricalTransformer(ColumnTransformer):
    """
    Data transformer for a categorical column.
    """
    dtype = DataType.categorical
    pd_dtype = "string"

    def __init__(self):
        super().__init__()
        self.oe = OneHotEncoder(sparse_output=False)

    params = (set(inspect.signature(ColumnTransformer).parameters) |
              set(inspect.signature(__init__).parameters)) - {"self"}

    def _fit(self, data: pd.Series):
        self.oe.fit(data.values.reshape((-1, 1)))

    def _standardize(self, data: pd.Series) -> pd.DataFrame:
        return data.to_frame("val")

    def inverse_standardize(self, data: pd.DataFrame) -> pd.Series:
        return data["val"]

    @property
    def standardized_types(self) -> Dict[str, DataType]:
        return {"val": DataType.categorical}

    def _normalize(self, data: pd.Series) -> pd.DataFrame:
        result = self.oe.transform(data.values.reshape((-1, 1)))
        result = pd.DataFrame(result, columns=[f"cat-{i:02d}" for i in range(result.shape[1])], index=data.index)
        return result

    def inverse_normalize(self, data: pd.DataFrame) -> pd.Series:
        result = self.oe.inverse_transform(data.values)
        return pd.Series(result[:, 0], index=data.index)

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        return [(len(self.oe.categories_[0]), SpanType.discrete)]

    @property
    def normalized_columns(self) -> List[str]:
        return [f"cat-{i:02d}" for i in range(len(self.oe.categories_[0]))]


class NumericTransformer(ColumnTransformer):
    """
    Data transformer for a numeric column.
    """
    dtype = DataType.numeric
    pd_dtype = "float"

    def __init__(self, n_bins: int = 20):
        """
        Parameters
        ----------
        n_bins : int
            The guiding KMeans number of bins. If the value is non-positive, no guiding bins will be applied.
        """
        super().__init__()
        self.kmeans = KBinsDiscretizer(n_bins=n_bins, strategy="kmeans", encode="onehot-dense") if n_bins > 0 else None
        self.scaler = StandardScaler()

    params = (set(inspect.signature(ColumnTransformer).parameters) |
              set(inspect.signature(__init__).parameters)) - {"self"}

    def _fit(self, data: pd.Series):
        fit_kmeans = self.kmeans is not None and data.max() - data.min() > 1e-6
        data = data.values.reshape((-1, 1))
        if fit_kmeans:
            self.kmeans.fit(data)
        else:
            self.kmeans = None
        self.scaler.fit(data)

    def _standardize(self, data: pd.Series) -> pd.DataFrame:
        return data.to_frame("val")

    def inverse_standardize(self, data: pd.DataFrame) -> pd.Series:
        return data["val"]

    @property
    def standardized_types(self) -> Dict[str, DataType]:
        return {"val": DataType.numeric}

    def _normalize(self, data: pd.Series) -> pd.DataFrame:
        values = data.values.reshape((-1, 1))
        if self.kmeans is None:
            bin_data = pd.DataFrame(index=data.index)
        else:
            bin_data = self.kmeans.transform(values)
            bin_data = pd.DataFrame(
                bin_data, columns=[f"bin-{i:02d}" for i in range(bin_data.shape[1])], index=data.index
            )
        scaled_data = self.scaler.transform(values)[:, 0]
        scaled_data = pd.Series(scaled_data, index=data.index).to_frame("val")
        return pd.concat([bin_data, scaled_data], axis=1)

    def inverse_normalize(self, data: pd.DataFrame) -> pd.Series:
        scaled = data[["val"]].values
        recovered = self.scaler.inverse_transform(scaled)[:, 0]
        return pd.Series(recovered, index=data.index)

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        if self.kmeans is None:
            return [(1, SpanType.continuous)]
        else:
            return [(self.kmeans.n_bins_[0], SpanType.discrete), (1, SpanType.continuous)]

    @property
    def normalized_columns(self) -> List[str]:
        names = []
        if self.kmeans is not None:
            names = [f"bin-{i:02d}" for i in range(self.kmeans.n_bins_[0])]
        names.append("val")
        return names


dt_components = Literal[
    "year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond",
    "month_name", "dayofweek", "day_name", "day_to_month_end", "am_pm", "is_month_start", "is_month_end", "is_holiday"
]
holiday_calendar = USFederalHolidayCalendar()
# holiday refers to US federal holiday


class DatetimeTransformer(ColumnTransformer):
    """
    Data transformer for a datetime column.
    """
    dtype = DataType.datetime
    pd_dtype = "datetime64[ns]"

    def __init__(
            self, auxiliary_components: List[dt_components] = [
                "year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond",
                "month_name", "dayofweek", "day_name", "day_to_month_end", "am_pm", "is_month_start", "is_month_end",
                "is_holiday"
            ],
            edit_by_components: List[dt_components] = [
                "year", "month_name", "day_name", "day", "day_to_month_end", "is_month_start", "is_month_end",
                "hour", "am_pm", "minute", "second", "millisecond", "microsecond", "nanosecond"
            ],
            **kwargs
    ):
        """
        Parameters
        ----------
        auxiliary_components : List[dt_components]
            The list of auxiliary columns to describe the datetime. When columns' order in the corresponding network
            matters, the components' order in this parameter describes that order. Components that maintains constant
            in the column will be omitted from the transformed result.
        edit_by_components : List[dt_components]
            The list of components to make sure the recovered datetime value must follow. For example, if the recovered
            date is 2020-01-03 but is_month_start is true and provided in this list, the value will be converted to the
            start day of that month, which is 2020-01-01. The order of components in this list are the edition order,
            so the edition in latter components may override the values in the previous components.
        **kwargs
            Arguments to numeric transformer for numeric components.
        """
        super().__init__()
        self.auxiliary_components = auxiliary_components
        self.edit_by_components = edit_by_components
        if not set(edit_by_components) <= set(auxiliary_components):
            raise ValueError("Edit components must be a subset of all auxiliary components.")
        self._kwargs = kwargs

        self.components = {}
        self.core_transformer = NumericTransformer(**kwargs)
        self._mean = None
        self._holidays = None

    params = (set(inspect.signature(ColumnTransformer).parameters) |
              set(inspect.signature(CategoricalTransformer.__init__).parameters) |
              set(inspect.signature(NumericTransformer.__init__).parameters) |
              set(inspect.signature(__init__).parameters)) - {"self", "kwargs"}

    def _fit(self, data: pd.Series):
        self._mean = data.mean()
        self._holidays = holiday_calendar.holidays(start=data.min(), end=data.max())
        for component in self.auxiliary_components:
            comp_values = self._get_component(data, component)
            if comp_values.nunique() == 1:
                self.components[component] = comp_values.iloc[0]
            elif pd.api.types.is_numeric_dtype(comp_values.dtype):
                transformer = NumericTransformer(**self._kwargs)
                transformer.fit(comp_values)
                self.components[component] = transformer
            else:
                transformer = CategoricalTransformer()
                transformer.fit(comp_values)
                self.components[component] = transformer
        num_data = (data - self._mean).dt.total_seconds()
        self.core_transformer.fit(num_data)

    def _get_component(self, data: pd.Series, component: dt_components) -> pd.Series:
        if component in [
            "year", "month", "day", "hour", "minute", "second", "nanosecond", "dayofweek",
            "is_month_start", "is_month_end"
        ]:
            result = getattr(data.dt, component)
            if component.startswith("is"):
                result = result.astype("string")
            return result
        elif component == "millisecond":
            return data.dt.microsecond // 1000
        elif component == "microsecond":
            return data.dt.microsecond % 1000
        elif component in ["month_name", "day_name"]:
            return getattr(data.dt, component)()
        elif component == "day_to_month_end":
            return data.dt.daysinmonth - data.dt.day
        elif component == "am_pm":
            return data.dt.hour.apply(lambda x: "am" if x < 12 else "pm")
        elif component == "is_holiday":
            return data.dt.date.astype("datetime64[ns]").isin(self._holidays).astype("string")
        else:
            raise ValueError(f"Component {component} is not recognized.")

    def _standardize(self, data: pd.Series) -> pd.DataFrame:
        result = {}
        for component_name, transformer in self.components.items():
            if isinstance(transformer, ColumnTransformer):
                result[component_name] = transformer.standardize(self._get_component(data, component_name))
        num_data = (data - self._mean).dt.total_seconds()
        result["val"] = self.core_transformer.standardize(num_data)
        combined = pd.concat(result, axis=1)
        combined = flatten_columns(combined)
        return combined

    def inverse_standardize(self, data: pd.DataFrame) -> pd.Series:
        num_data = extract(data, "val")
        diff_sec = self.core_transformer.inverse_standardize(num_data)
        diff_sec = pd.to_timedelta(diff_sec, unit="s")
        dat_data = self._mean + diff_sec

        for component_name in self.edit_by_components:
            transformer = self.components[component_name]
            if isinstance(transformer, ColumnTransformer):
                component_data = extract(data, component_name)
                component_data = transformer.inverse_standardize(component_data)
            else:
                component_data = transformer
            dat_data = self._edit_component(dat_data, component_name, component_data)
        return dat_data

    def _edit_component(
            self, data: pd.Series, component: dt_components, component_data: Union[pd.Series, float, int, str]
    ) -> pd.Series:
        if component == "is_holiday":
            raise ValueError("Component 'is_holiday' is not supported as basis of edition.")
        if component == "month_name":
            month_name_map = {
                "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
            }
            if isinstance(component_data, pd.Series):
                component_data = component_data.replace(month_name_map)
            else:
                component_data = month_name_map[component_data]
            component = "month"
        elif component == "millisecond":
            original_component = self._get_component(data, "microsecond")
            component_data = component_data * 1000 + original_component
            component = "microsecond"
        elif component == "microsecond":
            original_component = self._get_component(data, "millisecond")
            component_data = original_component * 1000 + component_data
        elif component == "day_to_month_end":
            days_in_month = data.dt.daysinmonth
            component_data = days_in_month - component_data
            component = "day"
        elif component == "am_pm":
            if isinstance(component_data, pd.Series):
                hours = data.dt.hour
                is_am = component_data == "am"
                hours[is_am] = hours[is_am].clip(upper=11)
                hours[~is_am] = hours[~is_am].clip(lower=12)
                component_data = hours
            elif component_data == "am":
                component_data = data.dt.hour.clip(upper=11)
            elif component_data == "pm":
                component_data = data.dt.hour.clip(lower=12)
            else:
                raise ValueError(f"Component am_pm value {component_data} is invalid.")
            component = "hour"
        elif component == "is_month_start":
            if isinstance(component_data, pd.Series):
                days = data.dt.day
                is_month_start = component_data == "True"
                days[is_month_start] = 1
                days[~is_month_start] = days[~is_month_start].clip(lower=2)
                component_data = days
            elif component_data == "True":
                component_data = 1
            elif component_data == "False":
                component_data = data.dt.day.clip(lower=2)
            else:
                raise ValueError(f"Component is_month_start value {component_data} is invalid.")
            component = "day"
        elif component == "is_month_end":
            if isinstance(component_data, pd.Series):
                days = data.dt.day
                days_in_month = data.dt.daysinmonth
                is_month_end = component_data == "True"
                days[is_month_end] = days_in_month
                days[~is_month_end] = days.clip(upper=days_in_month - 1)
                component_data = days
            elif component_data == "True":
                component_data = data.dt.daysinmonth
            elif component_data == "False":
                days = data.dt.day
                days_in_month = data.dt.daysinmonth
                component_data = days.clip(upper=days_in_month - 1)
            else:
                raise ValueError(f"Component is_month_end value {component_data} is invalid.")
            component = "day"
        elif component == "day_name":
            day_name_map = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6
            }
            if isinstance(component_data, pd.Series):
                component_data = component_data.replace(day_name_map)
            else:
                component_data = day_name_map[component_data]
            component = "dayofweek"

        if component in ["year", "month", "day", "hour", "minute", "second", "microsecond", "nanosecond"]:
            if isinstance(component_data, pd.Series):
                if component == "year":
                    component_data = component_data.clip(1900, 2100)
                elif component == "month":
                    component_data = component_data.clip(1, 12)
                elif component == "day":
                    component_data = component_data.clip(1, data.dt.daysinmonth)
                elif component == "hour":
                    component_data = component_data.clip(0, 23)
                elif component == "minute":
                    component_data = component_data.clip(0, 59)
                elif component == "second":
                    component_data = component_data.clip(0, 59)
                elif component == "millisecond":
                    component_data = component_data.clip(0, 1000)
                elif component == "microsecond":
                    component_data = component_data.clip(0, 1000)
                elif component == "nanosecond":
                    component_data = component_data.clip(0, 1000)
                component_data = component_data.round().astype(np.int32)
                if component in {"month", "year"}:
                    new_date = pd.to_datetime(pd.DataFrame(
                        {"year": data.dt.year, "month": data.dt.month, "day": 1} | {component: component_data}
                    ))
                    data = pd.DataFrame({
                        "dat": data, "comp": component_data,
                        "day": pd.DataFrame({"m": new_date.dt.daysinmonth, "a": data.dt.day}).min(axis=1)
                    }).apply(
                        lambda row: row["dat"].replace(**{component: row["comp"], "day": row["day"]}), axis=1
                    )
                else:
                    data = pd.DataFrame({"dat": data, "comp": component_data}).apply(
                        lambda row: row["dat"].replace(**{component: row["comp"]}), axis=1
                    )
            else:
                data = data.apply(lambda x: x.replace(**{component: component_data}))
        elif component == "dayofweek":
            original_component = data.dt.dayofweek
            addition = (component_data - original_component + 7) % 7
            subtraction = (component_data - original_component - 7) % 7
            day_diff = pd.DataFrame({"add": addition, "sub": subtraction}).apply(
                lambda row: row["add"] if abs(row["add"]) < abs(row["sub"]) else row["sub"], axis=1
            )
            day_diff = pd.to_timedelta(day_diff, unit="d")
            data = data + day_diff
        else:
            raise ValueError(f"Component {component} is not editable.")
        return data

    @property
    def standardized_types(self) -> Dict[str, DataType]:
        result = {}
        for component_name, transformer in self.components.items():
            if isinstance(transformer, ColumnTransformer):
                for k, v in transformer.standardized_types.items():
                    result[f"{component_name}.{k}"] = v
        for k, v in self.core_transformer.standardized_types.items():
            result[f"val.{k}"] = v
        return result

    def _normalize(self, data: pd.Series) -> pd.DataFrame:
        result = {}
        for component_name, transformer in self.components.items():
            if isinstance(transformer, ColumnTransformer):
                result[component_name] = transformer.normalize(self._get_component(data, component_name))
        num_data = (data - self._mean).dt.total_seconds()
        result["val"] = self.core_transformer.normalize(num_data)
        combined = pd.concat(result, axis=1)
        combined = flatten_columns(combined)
        return combined

    def inverse_normalize(self, data: pd.DataFrame) -> pd.Series:
        num_data = extract(data, "val")
        diff_sec = self.core_transformer.inverse_normalize(num_data)
        diff_sec = pd.to_timedelta(diff_sec, unit="s")
        dat_data = self._mean + diff_sec

        for component_name in self.edit_by_components:
            transformer = self.components[component_name]
            if isinstance(transformer, ColumnTransformer):
                component_data = extract(data, component_name)
                component_data = transformer.inverse_normalize(component_data)
            else:
                component_data = transformer
            dat_data = self._edit_component(dat_data, component_name, component_data)
        return dat_data

    @property
    def spans(self) -> List[Tuple[int, SpanType]]:
        results = []
        for component_name, transformer in self.components.items():
            if isinstance(transformer, ColumnTransformer):
                results.extend(transformer.spans)
        results.extend(self.core_transformer.spans)
        return results

    @property
    def normalized_columns(self) -> List[str]:
        names = []
        for component_name, transformer in self.components.items():
            if isinstance(transformer, ColumnTransformer):
                for c in transformer.normalized_columns:
                    names.append(f"{component_name}.{c}")
        for c in self.core_transformer.normalized_columns:
            names.append(f"val.{c}")
        return names


column_transformers = {
    DataType.categorical: CategoricalTransformer,
    DataType.numeric: NumericTransformer,
    DataType.datetime: DatetimeTransformer,
}
