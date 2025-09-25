import json
import os
from typing import Dict, Tuple

import pandas as pd
import torch
from arfpy import arf
import pickle
from sklearn.preprocessing import LabelEncoder

from .base import StaticGenerator
from ..dataset import DataType


class ARFGenerator(StaticGenerator):
    def __init__(self, output_dir: str, **kwargs):
        super().__init__(output_dir)
        self.encoders = {}
        self.categorical_columns = []
        self.n_left = 0

    def _encode_categorical_data(self, data: pd.DataFrame, data_types: Dict[str, DataType], fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder"""
        data_encoded = data.copy()
        
        for col, dtype in data_types.items():
            if col in data.columns and dtype == DataType.categorical:
                if fit:
                    # Fit encoder during training
                    self.encoders[col] = LabelEncoder()
                    data_encoded[col] = self.encoders[col].fit_transform(data[col].astype(str))
                    self.categorical_columns.append(col)
                else:
                    # Transform during generation (encoder already fitted)
                    if col in self.encoders:
                        # Handle unseen categories by using the most frequent class
                        unseen_mask = ~data[col].astype(str).isin(self.encoders[col].classes_)
                        if unseen_mask.any():
                            data_temp = data[col].astype(str).copy()
                            data_temp[unseen_mask] = self.encoders[col].classes_[0]  # Use first class for unseen
                            data_encoded[col] = self.encoders[col].transform(data_temp)
                        else:
                            data_encoded[col] = self.encoders[col].transform(data[col].astype(str))
        
        return data_encoded

    def _decode_categorical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Decode categorical variables back to original labels"""
        data_decoded = data.copy()
        
        for col in self.categorical_columns:
            if col in data.columns and col in self.encoders:
                # Round to nearest integer for categorical decoding
                encoded_values = data[col].round().astype(int)
                # Clip values to valid range
                encoded_values = encoded_values.clip(0, len(self.encoders[col].classes_) - 1)
                data_decoded[col] = self.encoders[col].inverse_transform(encoded_values)
        
        return data_decoded

    def train(
            self, data: pd.DataFrame, agg_data: pd.DataFrame,
            data_types: Dict[str, DataType], agg_data_types: Dict[str, DataType], **kwargs
    ):
        self.n_left = data.shape[-1]
        
        # # Encode categorical variables
        data_encoded = self._encode_categorical_data(data, data_types, fit=True)
        agg_data_encoded = self._encode_categorical_data(agg_data, agg_data_types, fit=True)
        
        # Combine data
        combined_data = pd.concat([data_encoded, agg_data_encoded], axis=1)
        
        # Train ARF model
        my_arf = arf.arf(x=combined_data)
        
        # Save model and metadata
        with open(os.path.join(self.output_dir, "model.pkl"), "wb") as f:
            pickle.dump(my_arf, f)
        
        with open(os.path.join(self.output_dir, "encoders.pkl"), "wb") as f:
            pickle.dump(self.encoders, f)
            
        with open(os.path.join(self.output_dir, "info.json"), "w") as f:
            json.dump({
                "n_left": self.n_left,
                "categorical_columns": self.categorical_columns
            }, f)

    def generate(self, n_samples: int, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load model and metadata
        with open(os.path.join(self.output_dir, "model.pkl"), "rb") as f:
            my_arf = pickle.load(f)
            
        with open(os.path.join(self.output_dir, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)
            
        with open(os.path.join(self.output_dir, "info.json"), "r") as f:
            info = json.load(f)
            self.n_left = info["n_left"]
            self.categorical_columns = info["categorical_columns"]
        
        # Generate samples
        my_arf.forde()
        out = my_arf.forge(n=n_samples)
        
        # Split data
        data_part = out.iloc[:, :self.n_left]
        agg_part = out.iloc[:, self.n_left:]
        
        # Decode categorical variables
        data_part_decoded = self._decode_categorical_data(data_part)
        agg_part_decoded = self._decode_categorical_data(agg_part)
        
        return data_part_decoded, agg_part_decoded

    def generate_from_static(self, static: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Partial generation is not supported for ARF.")
