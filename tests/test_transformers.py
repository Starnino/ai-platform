import unittest
import tempfile
import textwrap
import sys
import types
import os
import numpy as np

from src.core.transformers.loader import load_transformers
from src.shared.loader import load_yaml
from src.core.transformers.base import BasePreprocessor, BasePostprocessor, Transformer


def register_fake_modules():
    """Registra moduli fake in sys.modules così import_class li può risolvere."""

    # src.transformers.standard
    mod_standard = types.ModuleType("src.transformers.standard")

    class MultiplyPreprocessor(BasePreprocessor):
        """Moltiplica i dati per un fattore dinamico (default=1.0)."""
        def __init__(self, file_path: str = ""):
            super().__init__(file_path=file_path)
            self.file_path = file_path
            self.last_kwargs = {}

        def preprocess(self, data, context=None, **kwargs):
            self.last_kwargs = dict(kwargs)
            factor = kwargs.get("factor", 1.0)
            return np.array(data) * factor

    class AddPreprocessor(BasePreprocessor):
        """Somma un offset dinamico (default=0.0)."""
        def __init__(self, note: str = ""):
            super().__init__(note=note)
            self.note = note
            self.last_kwargs = {}

        def preprocess(self, data, context=None, **kwargs):
            self.last_kwargs = dict(kwargs)
            offset = kwargs.get("offset", 0.0)
            return np.array(data) + offset

    class ClipperPostprocessor(BasePostprocessor):
        """Clippa i valori al di sotto di lower_bound (default=0.0)."""
        def __init__(self):
            super().__init__()
            self.last_kwargs = {}

        def postprocess(self, data, context=None, **kwargs):
            self.last_kwargs = dict(kwargs)
            lb = kwargs.get("lower_bound", 0.0)
            arr = np.array(data)
            arr[arr < lb] = lb
            return arr

    mod_standard.MultiplyPreprocessor = MultiplyPreprocessor
    mod_standard.AddPreprocessor = AddPreprocessor
    mod_standard.ClipperPostprocessor = ClipperPostprocessor

    sys.modules["src.transformers.standard"] = mod_standard


class TestStandardPreprocessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_fake_modules()

    def _write_yaml(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(text))
        return path

    def test_preprocessors_and_postprocessor(self):
        yaml_text = """
        - name: DemoTransformer
          preprocessors:
            - name: MultiplyPreprocessor
              src: src.transformers.standard
              parameters:
                file_path: dummy.pkl
                factor: {type: float, default: 2.0, min: 0.5, max: 10.0}
            - name: AddPreprocessor
              src: src.transformers.standard
              parameters:
                offset: {type: float, default: 1.0}
          postprocessors:
            - name: ClipperPostprocessor
              src: src.transformers.standard
              parameters:
                lower_bound: {type: float, default: 0.0}
        """
        path = self._write_yaml(yaml_text)
        try:
            config = load_yaml(path)
            models = load_transformers(config)
            t: Transformer = models["DemoTransformer"]

            # input
            x = np.array([1.0, -1.0, 3.0])
            # preprocess: x * 2 + 1
            pre = t.preprocess(x)
            np.testing.assert_allclose(pre, np.array([3.0, -1.0, 7.0]))

            # postprocess: clip a 0
            out = t.postprocess(pre)
            np.testing.assert_allclose(out, np.array([3.0, 0.0, 7.0]))

            # parameters() summary
            params = t.summary()
            self.assertIn("MultiplyPreprocessor", params["preprocessors"])
            self.assertIn("AddPreprocessor", params["preprocessors"])
            self.assertIn("ClipperPostprocessor", params["postprocessors"])
        finally:
            os.remove(path)

    def test_required_param_missing(self):
        yaml_text = """
        - name: BadTransformer
          preprocessors:
            - name: MultiplyPreprocessor
              src: src.transformers.standard
              parameters:
                file_path: dummy.pkl
                factor: {type: float, required: true}
        """
        path = self._write_yaml(yaml_text)
        try:
            config = load_yaml(path)
            models = load_transformers(config)
            t: Transformer = models["BadTransformer"]
            x = np.array([1.0, 2.0])
            # factor è required → se non lo passo a runtime solleva
            with self.assertRaises(ValueError):
                t.preprocess(x)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
