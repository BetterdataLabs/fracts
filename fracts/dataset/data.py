import os
import shutil
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TSData:
    """
    Timeseries data.
    """
    def __init__(self, data_dir: str, static_data_path: Optional[str] = None, static_id: Optional[str] = None,):
        """
        Parameters
        ----------
        data_dir : str
            The path where all timeseries data are saved. Files are named `[STATIC_ID].csv`, and timeseries of each
            static ID is saved in a separate file. By static ID, we mean the name of the timeseries. For example,
            in the timeseries of the transactions of different customers, the customer ID is usually the static ID,
            and in stock data, the corporation ID is usually the static ID.
        static_data_path : str, optional
            The path where the static components of the timeseries data are saved. Each row in the file is a static
            description of a timeseries of a specific static ID. For example, in the timeseries of the transactions of
            different customers, this file should describe the profile of each customer, and in stock data, this file
            should provide a high-level overview of the corresponding corporation. This should also be a csv file.
            If it is not provided, it means the dataset does not have static component.
        static_id : str, optional
            When `static_data_path` is provided, this value means the column name of the static ID.
        """
        self.data_dir = data_dir
        self.static_ids = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                self.static_ids.append(filename[:-4])
        self.static_data = None
        self.static_data_path = static_data_path
        if static_data_path is not None:
            self.static_data = pd.read_csv(static_data_path, index_col=static_id).T
            self.static_data = self.static_data.set_axis([str(c) for c in self.static_data.columns], axis=1)
            self.static_data = self.static_data[self.static_ids]

    def __getitem__(self, item: Union[int, str]) -> Tuple[str, pd.DataFrame, pd.Series]:
        """
        Getting timeseries by static ID. If an integer is given, we get the i-th static ID's content, and if a string
        is given, the string is the static ID.

        Returns
        -------
        str
            The static ID.
        pd.DataFrame
            The timeseries data (temporal component).
        pd.Series
            The static columns' values.
        """
        if not isinstance(item, str):
            item = self.static_ids[item]
        if self.static_data is None:
            static = pd.Series(name=item)
        else:
            static = self.static_data[item]
        return item, pd.read_csv(self._get_path_for(item)), static

    def _get_path_for(self, static_id: str) -> str:
        data_path = os.path.join(self.data_dir, f"{static_id}.csv")
        if os.path.islink(data_path):
            data_path = os.readlink(data_path)
        return data_path

    def get_batch(self, indices: Union[slice, List[int]]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Get a batch of timeseries data.

        Parameters
        ----------
        indices : Union[slice, List[int]]
            The indices of the timeseries data to get.

        Returns
        -------
        Dict[str, pd.DataFrame]
            The timeseries data (temporal component). Keys are static IDs.
        pd.DataFrame
            The static columns' values. The index is the static ID.
        """
        static_ids = self.static_ids[indices] if isinstance(indices, slice) else np.array(self.static_ids)[indices]
        if self.static_data is None:
            static = pd.DataFrame(index=static_ids)
        else:
            static = self.static_data[static_ids].T
        ts = {}
        for static_id in static_ids:
            ts[static_id] = pd.read_csv(self._get_path_for(static_id))
        return ts, static

    def __len__(self) -> int:
        """
        Number of timeseries in the dataset.
        """
        return len(self.static_ids)

    def __iter__(self) -> Iterator[Tuple[str, pd.DataFrame, pd.Series]]:
        for i in range(len(self)):
            yield self[i]

    def split(self, test_size: float = 0.2, stratify: Optional[str] = None,
              out_path: Optional[str] = None) -> Tuple["TSData", "TSData"]:
        """
        Split the dataset for training and test sets.

        Parameters
        ----------
        test_size : float
            The proportion of test set.
        stratify : str, optional
            The column name for stratified split if applicable. This column should be a column in static data.
        out_path : str, optional
            The output path. If not provided, we will use the current directory of dataset.

        Returns
        -------
        TSData
            Training dataset.
        TSData
            Test dataset.
        """
        if self.static_data is None:
            static_data = None
        else:
            static_data = self.static_data.T
        if stratify is not None:
            stratify = static_data[stratify]

        if test_size > 0:
            train_static_ids, test_static_ids = train_test_split(
                self.static_ids, test_size=test_size, stratify=stratify
            )
        else:
            train_static_ids = self.static_ids
            test_static_ids = []

        splits = {
            "train": train_static_ids, "test": test_static_ids
        }
        outputs = {}
        data_dir = out_path if out_path is not None else self.data_dir
        os.makedirs(data_dir, exist_ok=True)
        for split_name, split_static_ids in splits.items():
            if os.path.exists(os.path.join(data_dir, split_name)):
                shutil.rmtree(os.path.join(data_dir, split_name))
            os.makedirs(os.path.join(data_dir, split_name))
            for static_id in split_static_ids:
                os.symlink(
                    os.path.join(self.data_dir, f"{static_id}.csv"),
                    os.path.join(data_dir, split_name, f"{static_id}.csv")
                )
            data = TSData(os.path.join(data_dir, split_name))
            if self.static_data is not None:
                data.static_data = self.static_data[split_static_ids]
            outputs[split_name] = data

        return outputs["train"], outputs["test"]


