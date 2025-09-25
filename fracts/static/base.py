import inspect
import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd

from ..dataset import DataType


class StaticGenerator(ABC):
    """Static data generator, which can be any typical tabular data generator."""
    def __init__(self, output_dir: str):
        """
        Parameters
        ----------
        output_dir : str
            The output directory where the model is saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    params = set(inspect.signature(__init__).parameters) - {"self"}

    @abstractmethod
    def train(self, data: pd.DataFrame, agg_data: pd.DataFrame,
              data_types: Dict[str, DataType], agg_data_types: Dict[str, DataType], **kwargs):
        """
        Train the static data generator. The model should be saved under output directory after training.

        Parameters
        ----------
        data : pd.DataFrame
            The raw data to train the generator on.
        agg_data : pd.DataFrame
            The raw aggregated data to train the generator on.
        data_types : Dict[str, DataType]
            The data types of each column.
        agg_data_types : Dict[str, DataType]
            The data types of each aggregated column.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic static values. Model should be reloaded before generation.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.
        **kwargs
            Model-specific arguments.

        Returns
        -------
        pd.DataFrame
            The synthetically generated data.
        pd.DataFrame
            The synthetically generated aggregated part data.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_from_static(self, static: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic aggregated values from static values. Model should be reloaded before generation.

        Parameters
        ----------
        static : pd.DataFrame
            The static values.
        **kwargs
            Model-specific arguments.

        Returns
        -------
        pd.DataFrame
            The synthetically generated aggregated part of data.
        """
        raise NotImplementedError()
