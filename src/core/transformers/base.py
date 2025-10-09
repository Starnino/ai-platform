from abc import ABC, abstractmethod
from typing import List, Any, Dict
import logging
import numpy as np
from .helpers import _check_value

logger = logging.getLogger(__name__)

class TransformerContext:
    """
    Class to manage the context for transformers, including input data, output data, and a key-value store.
    """
    def __init__(self):
        self._store = {}

    def set(self, key: str, value):
        self._store[key] = value

    def get(self, key: str, default=None):
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def all(self):
        return list(self._store.keys())

    def after_preprocessors(self, data: np.ndarray):
        """Optional: called by Transformer after all preprocessing steps."""
        pass
    
    def before_postprocessors(self, data: np.ndarray):
        """Optional: called after model prediction, before postprocessors."""
        pass

class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        self._name = self.__class__.__name__
        self._static_params: Dict[str, Any] = kwargs
        self._dynamic_params: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value) if value is not None else self.__class__.__name__
    
    @property
    def static_params(self) -> Dict[str, Any]:
        return self._static_params

    @property
    def dynamic_params(self) -> Dict[str, Any]:
        return self._dynamic_params

    @dynamic_params.setter
    def dynamic_params(self, params: Dict[str, Any]):
        self._dynamic_params = params

    def validate_params(self, params: Dict[str, Any] | None) -> Dict[str, Any]:
        params = params or {}
        validated = {}
        for k, spec in self._dynamic_params.items():
            if k in params:
                try:
                    validated[k] = _check_value(params[k], spec, path=k)
                except ValueError as e:
                    logger.error(f"Invalid parameter '{k}' for preprocessor {self.name}: {e}")
                    raise ValueError(f"Invalid parameter '{k}': {e}") from e
            elif "default" in spec:
                validated[k] = spec["default"]
            elif spec.get("required"):
                raise ValueError(f"Missing required parameter: {k}")
        return validated

class BasePreprocessor(BaseProcessor):
    @abstractmethod
    def preprocess(self, data, context=None, **kwargs) -> np.ndarray:
        pass

class BasePostprocessor(BaseProcessor):
    @abstractmethod
    def postprocess(self, data, context=None, **kwargs) -> Any:
        pass

class Transformer:
    def __init__(self, context: TransformerContext = None, preprocessors: List[BasePreprocessor] = None, postprocessors: List[BasePostprocessor] = None):
        self._context = context
        self._preprocessors = preprocessors or []
        self._postprocessors = postprocessors or []

    @property
    def context(self) -> TransformerContext:
        return self._context
    
    @property
    def preprocessors(self) -> List[BasePreprocessor]:
        return self._preprocessors

    @property
    def postprocessors(self) -> List[BasePostprocessor]:
        return self._postprocessors

    def preprocess(self, input_data, params_map: dict | None = None) -> np.ndarray:
        params_map = params_map or {}
        for preprocessor in self.preprocessors:
            dynamic_params = preprocessor.validate_params(params_map.get(preprocessor.name, {}))
            runtime_params = {**preprocessor.static_params, **dynamic_params}
            logger.info(f"Applying preprocessor: {preprocessor.name} with params: {runtime_params}")
            input_data = preprocessor.preprocess(input_data, context=self.context, **dynamic_params)
        if self.context:
            logger.info(f"Context after preprocessors: {self.context.all()}")
            self.context.after_preprocessors(input_data)
        return input_data

    def postprocess(self, output_data, params_map: dict | None = None) -> Any:
        params_map = params_map or {}
        if self.context:
            logger.info(f"Context before postprocessors: {self.context.all()}")
            self.context.before_postprocessors(output_data)
        for postprocessor in self.postprocessors:
            dynamic_params = postprocessor.validate_params(params_map.get(postprocessor.name, {}))
            runtime_params = {**postprocessor.static_params, **dynamic_params}
            logger.info(f"Applying postprocessor: {postprocessor.name} with params: {runtime_params}")
            output_data = postprocessor.postprocess(output_data, context=self.context, **dynamic_params)
        return output_data
    
    def summary(self) -> dict:
        def _normalize(d: dict | None) -> dict | None:
            return d if d else None

        def _summarize(processors):
            result = {}
            for p in processors:
                static = _normalize(p.static_params)
                dynamic = _normalize(p.dynamic_params)

                if static is None and dynamic is None:
                    result[p.__class__.__name__] = None
                else:
                    result[p.__class__.__name__] = {
                        "static": static,
                        "dynamic": dynamic,
                    }
            return result

        return {
            "preprocessors": _summarize(self.preprocessors),
            "postprocessors": _summarize(self.postprocessors),
        }