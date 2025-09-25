import json
import os
from typing import Dict, Tuple

import pandas as pd
import torch
from ctgan import CTGAN, TVAE

from .base import StaticGenerator
from ..dataset import DataType


class CTGANGenerator(StaticGenerator):
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir)
        self.model = CTGAN(cuda=torch.cuda.is_available(), **kwargs)
        self.n_left = 0

    def train(
            self, data: pd.DataFrame, agg_data: pd.DataFrame,
            data_types: Dict[str, DataType], agg_data_types: Dict[str, DataType], **kwargs
    ):
        self.n_left = data.shape[-1]
        data = pd.concat([data, agg_data], axis=1)
        self.model.fit(
            data, discrete_columns=[c for c, t in (data_types | agg_data_types).items() if t == DataType.categorical],
            **kwargs
        )
        self.model.save(os.path.join(self.output_dir, "model.pt"))
        with open(os.path.join(self.output_dir, "info.json"), "w") as f:
            json.dump({"n_left": self.n_left}, f)

    def generate(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with open(os.path.join(self.output_dir, "info.json"), "r") as f:
            info = json.load(f)
            self.n_left = info["n_left"]
        self.model = self.model.load(os.path.join(self.output_dir, "model.pt"))
        out = self.model.sample(n_samples)
        return out.iloc[:, :self.n_left], out.iloc[:, self.n_left:]

    def generate_from_static(self, static: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Partial generation is not supported for CTGAN.")


class TVAEGenerator(StaticGenerator):
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir)
        self.model = TVAE(cuda=torch.cuda.is_available(), **kwargs)
        self.n_left = 0

    def train(
            self, data: pd.DataFrame, agg_data: pd.DataFrame,
            data_types: Dict[str, DataType], agg_data_types: Dict[str, DataType], **kwargs
    ):
        self.n_left = data.shape[-1]
        data = pd.concat([data, agg_data], axis=1)
        self.model.fit(
            data, discrete_columns=[c for c, t in (data_types | agg_data_types).items() if t == DataType.categorical],
            **kwargs
        )
        self.model.save(os.path.join(self.output_dir, "model.pt"))
        with open(os.path.join(self.output_dir, "info.json"), "w") as f:
            json.dump({"n_left": self.n_left}, f)

    def generate(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with open(os.path.join(self.output_dir, "info.json"), "r") as f:
            info = json.load(f)
            self.n_left = info["n_left"]
        self.model = self.model.load(os.path.join(self.output_dir, "model.pt"))
        out = self.model.sample(n_samples)
        return out.iloc[:, :self.n_left], out.iloc[:, self.n_left:]

    def generate_from_static(self, static: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Partial generation is not supported for TVAE.")
