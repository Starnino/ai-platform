from src.core.transformers import BasePreprocessor
import numpy as np

class BatchPreprocessor(BasePreprocessor):
    def preprocess(self, data: list[dict], **kwargs) -> np.ndarray:
        """
        Preprocess input data for prediction.
        
        Parameters:
        ----------
        data : dict or pd.DataFrame
            Input data containing features
            
        Returns:
        -------
        np.ndarray
            Preprocessed data ready for model
        """
        def build_batch(batch: dict):
            feature_1d = []
            feature_2d = []

            # Separate features into 1D and 2D arrays
            for key, value in batch.items():
                arr = np.array(value)
                if arr.ndim == 1:
                    feature_1d.append(arr)  # shape: (timesteps,)
                elif arr.ndim == 2:
                    feature_2d.append(arr)  # shape: (timesteps, n_dim)
                else:
                    raise ValueError(f"Unsupported shape {arr.shape} for key '{key}'")
            
            rows = feature_1d.copy()
            if feature_2d:
                n_dim = feature_2d[0].shape[1]
                for n in range(n_dim):
                    for arr in feature_2d:
                        rows.append(arr[:, n])  # shape: (timesteps,)

            # (timesteps, n_features)
            return np.stack(rows, axis=1)

        X = np.stack([build_batch(d) for d in data]) # (batch_size, timesteps, features)
        return X