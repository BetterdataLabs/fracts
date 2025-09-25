from typing import List, Tuple

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import TSData

class TSDataset(Dataset):
    """
    One-time conversion of the whole TSData object to pinned float32 tensors.
    Afterwards __getitem__ is only an index lookup – no Python / pandas work.
    """
    def __init__(self, data: TSData, pin: bool = True):
        ids, statics, series = [], [], []

        for sid, ts_df, static_df in data:        # <-- one-off pass
            ids.append(sid)
            statics.append(torch.tensor(static_df.values,  dtype=torch.float32))
            series.append( torch.tensor(ts_df.values,     dtype=torch.float32))

        self.ids     = ids
        self.static  = torch.stack(statics)               #  [N, F_static]
        self.series  = series                             # ragged list
        if pin:
            self.static = self.static.pin_memory()
            self.series = [s.pin_memory() for s in self.series]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.static[idx], self.series[idx]

class TSInferenceDataset(Dataset):
    """
    Timeseries tensor dataset for PyTorch inference.
    """
    def __init__(self, static_data: pd.DataFrame, lengths: List[int]):
        """
        Parameters
        ----------
        static_data : pd.DataFrame
            Static data as conditions (normalized).
        lengths : List[int], optional
            Lengths of the time series.
        """
        self.static_data = static_data
        self.lengths = lengths

    def __len__(self) -> int:
        """
        Number of timeseries in the dataset.
        """
        return len(self.static_data)

    def __getitem__(self, item: int) -> Tuple[str, torch.Tensor]:
        """
        Getting a timeseries data condition.

        Parameters
        ----------
        item : int
            The index of the timeseries.

        Returns
        -------
        str
            The static ID.
        torch.Tensor
            The static conditions (1D), including aggregated values.
        """
        data = torch.from_numpy(self.static_data.iloc[item].values).float()
        length = self.lengths[item] if self.lengths is not None else None
        return self.static_data.index[item], data, length


class TSDataCollator:
    """
    Data collator for timeseries data.
    """
    def __init__(self, max_len: int):
        """
        Parameters
        ----------
        max_len : int
            The maximum length to pad all data to (L).
        """
        self.max_len = max_len

    def __call__(
            self, batch: List[Tuple[str, torch.Tensor, torch.Tensor]]
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wrap batches to tensors.

        Parameters
        ----------
        batch : List[Tuple[str, torch.Tensor, torch.Tensor]]
            The list of batches, where each batch is an item in `TSDataset`.

        Returns
        -------
        List[str]
            List of static ID names.
        torch.Tensor
            Static tensor, shape is (B, Ws).
        torch.Tensor
            Padded timeseries values tensor, shape is (B, L, Wt).
        torch.Tensor
            Length indicators, 0/1/-1 tensor where 1 means the last timestep position (before padding), 0 means other
            non-padding timestep positions, and -1 means padded positions. shape is (B, L).
        """
        static_ids = []
        static = []
        ts = []
        len_indicator = []
        for sid, s, t in batch:
            static_ids.append(sid)
            static.append(s)
            padded_t = F.pad(t, (0, 0, 0, self.max_len - t.shape[0]))
            ts.append(padded_t)
            this_len_indicator = F.one_hot(torch.tensor(t.shape[0] - 1), self.max_len)
            this_len_indicator[t.shape[0]:] = -1
            len_indicator.append(this_len_indicator)
        return static_ids, torch.stack(static), torch.stack(ts), torch.stack(len_indicator)


class TSInferenceDataCollator:
    """
    Data collator for timeseries data inference.
    """

    def __call__(
            self, batch: List[Tuple[str, torch.Tensor]]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Wrap batches to tensors.

        Parameters
        ----------
        batch : List[Tuple[str, torch.Tensor]]
            The list of batches, where each batch is an item in `TSInferenceDataset`.

        Returns
        -------
        List[str]
            List of static ID names.
        torch.Tensor
            Static tensor, shape is (B, Ws).
        """
        static_ids = []
        static = []
        lengths = []

        for sid, s, l in batch:
            static_ids.append(sid)
            static.append(s)
            if l is not None:
                lengths.append(l)
        return static_ids, torch.stack(static), lengths