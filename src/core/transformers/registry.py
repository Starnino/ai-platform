from collections.abc import Mapping
from .base import Transformer

class TransformerRegistry(Mapping):
    def __init__(self, transformers: dict[str, Transformer]):
        self._registry = transformers

    def __getitem__(self, key: str) -> Transformer:
        return self._registry[key]

    def __iter__(self):
        return iter(self._registry)

    def __len__(self):
        return len(self._registry)

    @property
    def all(self) -> dict[str, Transformer]:
        return dict(self._registry)