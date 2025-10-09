from collections.abc import Mapping
from .base import BaseModel

class ModelRegistry(Mapping):
    def __init__(self, models: dict[str, BaseModel]):
        self._registry = models

    def __getitem__(self, key: str) -> BaseModel:
        return self._registry[key]

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    @property
    def all(self) -> dict[str, BaseModel]:
        return dict(self._registry)

    def preload(self):
        for model in self._registry.values():
            if not model.is_loaded:
                model.load()