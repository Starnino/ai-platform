from .base import BasePreprocessor, BasePostprocessor
import sklearn.preprocessing as skp
import numpy as np
import sklearn.preprocessing as skp
from joblib import load
import os
import logging

logger = logging.getLogger(__name__)

class Dict2Array(BasePreprocessor):
    def preprocess(self, data: list[dict], **kwargs) -> np.ndarray:
        rows = []
        for d in data:
            # Ensure all values are lists and take the first element
            rows.append([float(v[0]) for v in d.values()])
        # Flatten the rows
        return np.array(rows, dtype=np.float32)
    
class StandardScaler(BasePreprocessor, BasePostprocessor):
    def __init__(self, file_path=None):
        super().__init__(file_path=file_path)
        self.file_path = file_path
        if not file_path or not os.path.exists(file_path):
            self.file_path = None
            logger.warning(f"File {file_path} does not exist. Initializing StandardScaler without loading.")
        self.scaler = skp.StandardScaler() if not self.file_path else load(self.file_path)

    def preprocess(self, data, **kwargs) -> np.ndarray:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if data.ndim > 2:
            batch_size, n_timesteps, n_features = data.shape
            data_2d = data.reshape(-1, n_features)  # (batch_size * n_timesteps, n_features)
            scaled = self.scaler.transform(data_2d) if self.file_path else self.scaler.fit_transform(data_2d)
            return scaled.reshape(batch_size, n_timesteps, n_features)
        else:            
            return self.scaler.transform(data) if self.file_path else self.scaler.fit_transform(data)
        
    def postprocess(self, data, **kwargs) -> dict:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1,1)
        if self.file_path:
            data = self.scaler.inverse_transform(data)
        else:
            raise ValueError("StandardScaler was not initialized, cannot inverse transform.")
        return data

class Clipper(BasePostprocessor):
    def __init__(self, lower_bound=None, upper_bound=None):
        super().__init__(lower_bound=lower_bound, upper_bound=upper_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def postprocess(self, data, **kwargs) -> dict:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        data = np.clip(data, self.lower_bound, self.upper_bound)
        return data