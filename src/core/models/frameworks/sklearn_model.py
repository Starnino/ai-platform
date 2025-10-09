"""
A specialized BaseModel for scikit-learn models or pipelines. 
By default, saving/loading uses joblib.

You should implement `model()` to create an untrained sklearn model 
(e.g., a Pipeline). Then override `preprocess(data: dict)` for input transformations 
if needed, or just pass if not used.
"""
import joblib
import sklearn
import logging
import numpy as np
from abc import abstractmethod
from ..base import BaseModel

logger = logging.getLogger(__name__)

class SklearnModel(BaseModel):
    def __init__(self, name: str, model_path: str, **kwargs):
        super().__init__(sklearn.base.BaseEstimator, name, model_path, framework='sklearn',**kwargs)

    @property
    def model_filename(self):
        return "model.joblib"

    @abstractmethod
    def model(self) -> sklearn.base.BaseEstimator:
        """
        Return an untrained sklearn model or pipeline.
        """
        pass

    def build(self):
        self._model = self.model()
        if not isinstance(self._model, self.model_type):
            raise RuntimeError(f"Model '{self.name}' is not a valid {self.model_type} instance.")
        logger.info(f"Model '{self.name}' built successfully.")

    def fit(self, X, y, **kwargs):
        if self._model is None:
            self.build()
        logger.info(f"Training model '{self.name}' on {len(X)} samples.")
        return self._model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        if self._model is None:
            self.build()
        logger.info(f"Predicting with model '{self.name}' on {len(X)} samples.")
        return self._model.predict(X)

    def evaluate(self, X, y, **kwargs):
        if self._model is None:
            self.build()
        logger.info(f"Evaluating model '{self.name}' on {len(X)} samples.")
        return self._model.score(X, y)

    def _save(self, path: str):
        joblib.dump(self._model, path)

    def _load(self, path: str):
        self._model = joblib.load(path)

    def summary(self) -> dict:
        if self._model is None:
            self.build()

        input_shape = str(tuple(-1, self._model.n_features_in_))

        try:
            X = np.random.rand(1, self._model.n_features_in_)
            y_pred = self.predict(X)
            output_shape = (-1, y_pred.shape[1]) if len(y_pred.shape) > 1 else (-1,)
        except:
            output_shape = None
            
        return {
            "name": self.name,
            "framework": self.framework,
            "version": self.version,
            "description": self.description,          
            "input_shape": input_shape,
            "output_shape": output_shape
        }