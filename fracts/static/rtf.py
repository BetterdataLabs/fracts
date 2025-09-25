import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from realtabformer import REaLTabFormer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import EncoderDecoderConfig
import time

from .base import StaticGenerator
from ..dataset import DataType


class RTFTabGenerator(StaticGenerator):
    def __init__(
            self, output_dir: str, **kwargs
    ):
        super().__init__(output_dir)
        self.model = REaLTabFormer(
            model_type="tabular",
            checkpoints_dir=os.path.join(output_dir, "ckpt"),
            samples_save_dir=os.path.join(output_dir, "samples"),
            **kwargs
        )
        self.n_left = 0

    def train(self, data: pd.DataFrame, agg_data: pd.DataFrame,
              data_types: Dict[str, DataType], agg_data_types: Dict[str, DataType], **kwargs):
        self.n_left = data.shape[-1]
        data = pd.concat([data, agg_data], axis=1)
        self.model.fit(data, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs)
        self.model.save(os.path.join(self.output_dir, "final"))
        with open(os.path.join(self.output_dir, "info.json"), "w") as f:
            json.dump({"n_left": self.n_left}, f)

    def generate(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        with open(os.path.join(self.output_dir, "info.json"), "r") as f:
            info = json.load(f)
            self.n_left = info["n_left"]

        model_path = sorted([
            p for p in (Path(self.output_dir) / "final").glob("id*") if p.is_dir()
        ], key=os.path.getmtime)[-1]

        self.model = REaLTabFormer.load_from_dir(model_path)
        out = self.model.sample(n_samples, **kwargs)
        return out.iloc[:, :self.n_left], out.iloc[:, self.n_left:]

    def generate_from_static(self, static: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Partial generation is not supported for REaLTabFormer tabular mode.")


class RTFRelGenerator(StaticGenerator):
    def __init__(
            self, output_dir: str, **kwargs
    ):
        super().__init__(output_dir)
        self.parent_model = REaLTabFormer(
            model_type="tabular",
            tabular_config=GPT2Config(n_embd=256, n_layer=6, n_head=8,n_inner=1024),
            checkpoints_dir=os.path.join(output_dir, "static-ckpt"),
            samples_save_dir=os.path.join(output_dir, "static-samples"),
            **kwargs
        )
        self.child_model: Optional[REaLTabFormer] = None
        self._args = kwargs

    def train(self, data: pd.DataFrame, agg_data: pd.DataFrame,
              data_types: Dict[str, DataType], agg_data_types: Dict[str, DataType], **kwargs):
        start = time.time()
        self.parent_model.fit(data, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs)
        print(f"Parent model training took {time.time() - start:.2f} seconds")
        self.parent_model.save(os.path.join(self.output_dir, "final-static"))
        parent_model_path = sorted([
            p for p in (Path(self.output_dir) / "final-static").glob("id*") if p.is_dir()
        ], key=os.path.getmtime)[-1]

        rel_conf = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=GPT2Config(n_embd=256, n_layer=6, n_head=8, n_inner=1024),
            decoder_config=GPT2Config(n_embd=256, n_layer=6, n_head=8, n_inner=1024)
        )

        print("training child model with parent model path")
        self.child_model = REaLTabFormer(
            model_type="relational",
            tabular_config=GPT2Config(n_embd=256, n_layer=6, n_head=8,n_inner=1024),
            relational_config=rel_conf,
            parent_realtabformer_path=parent_model_path,
            checkpoints_dir=os.path.join(self.output_dir, "agg-ckpt"),
            samples_save_dir=os.path.join(self.output_dir, "agg-samples"),
            output_max_length=None, **self._args
        )
        data = data.copy()
        data[".id"] = data.index
        agg_data = agg_data.copy()
        agg_data[".id"] = agg_data.index
        start = time.time()
        self.child_model.fit(df=agg_data, in_df=data, join_on=".id", **kwargs)
        print(f"Child model training took {time.time() - start:.2f} seconds")
        self.child_model.save(os.path.join(self.output_dir, "final-agg"))

    def generate(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        parent_model_path = sorted([
            p for p in (Path(self.output_dir) / "final-static").glob("id*") if p.is_dir()
        ], key=os.path.getmtime)
        
        if not parent_model_path:
            raise FileNotFoundError(f"No model directory starting with 'id' found in {os.path.join(self.output_dir, 'final-static')}")
        
        print(f"Loading parent model from: {parent_model_path[-1]}")
        self.parent_model = REaLTabFormer.load_from_dir(str(parent_model_path[-1]))
        static = self.parent_model.sample(n_samples, **kwargs)
        agg = self.generate_from_static(static, **kwargs)
        return static, agg

    def generate_from_static(self, static: pd.DataFrame, **kwargs) -> pd.DataFrame:
        child_model_path = sorted([
            p for p in (Path(self.output_dir) / "final-agg").glob("id*") if p.is_dir()
        ], key=os.path.getmtime)
        
        if not child_model_path:
            raise FileNotFoundError(f"No model directory starting with 'id' found in {os.path.join(self.output_dir, 'final-agg')}")
        
        print(f"Loading child model from: {child_model_path[-1]}")
        self.child_model = REaLTabFormer.load_from_dir(str(child_model_path[-1]))
        out = self.child_model.sample(
            input_unique_ids=static.index,
            input_df=static,
        )
        out = out.copy()
        
        out = out.replace('.', 0.0)
        out = out.apply(pd.to_numeric, errors='ignore')
        out = out.groupby(level=0).head(1)
        out.index = static.index  
        return out
