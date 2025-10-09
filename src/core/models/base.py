"""
Module: models.base.py

This module defines the abstract base class `BaseModel`, providing a standardized 
interface for building, training, evaluating, and saving machine learning models 
across various frameworks (e.g., TensorFlow, PyTorch, scikit-learn).

Class:
------
BaseModel(ABC)
    An abstract base class that requires subclasses to implement framework-specific 
    logic via a set of abstract methods,
    while offering common template methods (`build()`, `save()`, `load()`) for consistent
    usage and versioning.

Usage:
------
To create a custom model, subclass `BaseModel` and implement:

1. `build()`: Define the model architecture and compile it.
"""

from abc import ABC, abstractmethod
import os
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    def __init__(self, model_type, name: str, model_path: str, **kwargs):
        self.model_type = model_type
        self.name = name
        self.model_path = Path(model_path)
        self.framework = kwargs["framework"]
        self.description = kwargs.get("description", None)
        self.schema = kwargs.get("schema", None)
        version = kwargs.get("version", None)
        self._version = self._get_latest_version() if version in [None, "latest"] else int(version)
        self._model = None
        self._loaded = False

    @property
    @abstractmethod
    def model_filename(self) -> str:
        """
        Return the filename for saving/loading the model.
        This should be overridden by subclasses to specify the appropriate filename.
        """
        pass

    @property
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.
        Returns
        -------
        bool
            True if the model is loaded, False otherwise.
        """
        return self._loaded

    @property
    def version(self) -> int | None:
        return self._version
    
    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, X, y, **kwargs):
        pass
    
    def save(self, filepath: str | Path | None = None):
        """
        Save the model to a specified path or versioned directory. If `filepath` is None,
        versions are created automatically.
        Parameters
        ----------
        filepath : str, optional
            Path to save the model. If None, a versioned directory will be created.
        """
        if self._model is None:
            raise RuntimeError(f"Model '{self.name}' not built/trained.")

        if filepath is None:
            new_version = self._get_latest_version() + 1
            version_dir = self.model_path / str(new_version)
            version_dir.mkdir(parents=True, exist_ok=True)
            filepath = version_dir / self.model_filename
            self._version = new_version
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_save(filepath)

    def load(self, filepath: str | Path | None = None, version: int | str | None = None):
        """
        Load the model from a specified path or versioned directory. If `filepath` is None,
        the latest version is loaded. Otherwise, if `version` is specified, that version is loaded.
        Parameters
        ----------
        filepath : str, optional
            Path to load the model from. If None, the latest version will be loaded.
        version : int, optional
            Version number to load. If None, the latest version will be loaded.
        """
        if filepath is not None:
            path = Path(filepath)
            if not path.exists():
                logger.warning("File not found. Building new model.")
                self.build()
            else:
                self._load(path)
            self._loaded = True
            return

        # versioned path
        if version is not None:
            if version == "latest":
                self._version = self._get_latest_version()
            else:
                try:
                    self._version = int(version)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid version: {version}. Must be 'latest' or an integer.")
            
        path = self.model_path / str(self._version) / self.model_filename
        if not path.exists():
            logger.warning(f"Version {self._version} not found. Building new model.")
            self.build()
        else:
            self._load(path)
        logger.info(f"Model '{self.name}' loaded from {path}.")
        self._loaded = True

    def _atomic_save(self, path: Path):
        tmp = path.with_suffix(path.suffix)
        self._save(tmp)
        os.replace(tmp, path)

    @abstractmethod
    def _save(self, path: str | Path):
        pass

    @abstractmethod
    def _load(self, path: str | Path):
        pass

    @abstractmethod
    def summary(self) -> dict:
        """
        Return a summary of the model, including its name, type, and description.
        Returns
        -------
        str
            A summary string of the model.
        """
    
    def _get_latest_version(self) -> int:
        versions = []
        for d in self.model_path.iterdir():
            if d.is_dir() and d.name.isdigit():
                versions.append(int(d.name))
        return max(versions) if versions else 0