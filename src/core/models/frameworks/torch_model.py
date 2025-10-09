"""
A specialized BaseModel for PyTorch models.
This class expects the user to implement `model()` to define the actual PyTorch model architecture (e.g., a torch.nn.Module).
You should implement preprocess, fit, predict, and evaluate methods to handle data processing and model training/evaluation.
"""
import torch
import logging
from abc import abstractmethod
from ..base import BaseModel

logger = logging.getLogger(__name__)

class TorchModel(BaseModel):
    def __init__(self, name: str, model_path: str, device: str = None, **kwargs):
        super().__init__(torch.nn.Module, name, model_path, framework='pytorch', **kwargs)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module = None

    @property
    def model_filename(self):
        return "model.pt"
    
    def build(self):
        self._model = self.model().to(self.device)
        if not isinstance(self._model, self.model_type):
            raise RuntimeError(f"Model '{self.name}' is not a valid {self.model_type} instance.")
        logger.info(f"Model '{self.name}' built and moved to {self.device}.")

    @abstractmethod
    def model(self) -> torch.nn.Module:
        """
        Subclass must return an instance of torch.nn.Module.
        """
        pass

    @abstractmethod
    def fit(self, train_loader, **kwargs):
        """
        Subclass must implement the training loop.
        """
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        """
        Subclass must implement prediction logic.
        """
        pass

    @abstractmethod
    def evaluate(self, val_loader, **kwargs):
        """
        Subclass must implement evaluation logic.
        """
        pass

    def _save(self, path: str):
        torch.save(self._model.state_dict(), path)

    def _load(self, path: str):
        self._model = self.model()
        self._model.load_state_dict(torch.load(path, map_location=self.device))
        self._model.to(self.device)
    
    def summary(self) -> dict:
        # TODO : Implement a method to summarize the model architecture
        return {
            "name": self.name,
            "framework": self.framework,
            "version": self.version,
            "description": self.description,          
            "input_shape": None,
            "output_shape": None
        }