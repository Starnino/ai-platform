"""
Integration tests for a running gateway.
These tests require a running gateway with accessible backends.

The gateway URL and models config path can be set via environment variables:
- GATEWAY_URL: URL of the running gateway (default: http://localhost:8080)
- MODELS_CONFIG_PATH: Path to the models configuration YAML file (default: config/models.yaml)

The tests cover:
- Health check of the gateway
- For each model:
  - GET /models/{model_name} to retrieve model info
  - POST /models/{model_name}/predict with random valid input and check response
"""

import os
import random
import unittest
from typing import Any, Dict, List, Sequence
import httpx

def _rand_scalar(t):
    if t is float:
        return random.uniform(-20.0, 60.0)
    if t is int:
        return random.randint(0, 365)
    if t is bool:
        return bool(random.getrandbits(1))
    if t is str:
        return "sample"
    raise TypeError(f"Unsupported scalar type: {t}")

def _as_shape(seq) -> List[int]:
    if seq is None:
        return []
    if isinstance(seq, (list, tuple)):
        return [int(x) for x in seq]
    raise TypeError(f"Unexpected shape type: {type(seq)}")

def rand_by_shape(t, shape: Sequence[int]):
    shape = list(shape or [])
    if not shape:
        return _rand_scalar(t)
    size, *rest = shape
    return [rand_by_shape(t, rest) for _ in range(size)]

def build_instance(input_schema: List[Dict[str, Any]]) -> Dict[str, Any]:
    inst = {}
    for block in input_schema:
        t = block["type"]
        shape = _as_shape(block.get("shape"))
        inst[block["name"]] = rand_by_shape(t, shape)
    return inst

def matches_schema(value, t, shape: Sequence[int]) -> bool:
    shape = list(shape or [])
    if not shape:
        if t is bool:
            return isinstance(value, bool)
        return isinstance(value, t)
    if not isinstance(value, list) or len(value) != shape[0]:
        return False
    return all(matches_schema(v, t, shape[1:]) for v in value)


class TestPlatform(unittest.TestCase):
    """Integration tests for a running gateway (synchronous version)."""

    @classmethod
    def setUpClass(cls):
        random.seed(2025)

        cls.base_url = os.getenv("GATEWAY_URL", "http://localhost:8080")
        cls.yaml_path = os.getenv("MODELS_CONFIG_PATH", "config/models.yaml")

        from src.shared.loader import load_models_config
        cls.model_config_dict = load_models_config(cls.yaml_path)

        # Initialize sync httpx client
        cls.client = httpx.Client(base_url=cls.base_url, timeout=10.0)

        # Check that the gateway is up once
        try:
            r = cls.client.get("/health")
            if r.status_code != 200:
                print(f"Health check failed: {r.status_code} {r.text}")
                raise unittest.SkipTest(f"Health check failed at {cls.base_url}/health")
        except:
            print(f"Cannot connect to gateway at {cls.base_url}/health")
            raise unittest.SkipTest(f"Cannot connect to gateway at {cls.base_url}/health")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_end2end_models(self):
        """For each model, run /models and /predict."""
        for model_name, cfg in self.model_config_dict.items():
            r = self.client.get(f"/models/{model_name}")
            self.assertEqual(r.status_code, 200)

            input_schema = cfg.get("schema", {}).get("input", [])
            batch_size = 2
            instances = [build_instance(input_schema) for _ in range(batch_size)]

            r = self.client.post(f"/models/{model_name}/predict", json={"instances": instances})
            self.assertEqual(r.status_code, 200)
            js = r.json()
            self.assertIn("predictions", js)
            self.assertEqual(len(js["predictions"]), batch_size)

            output_schema = cfg.get("schema", {}).get("output", [])
            if output_schema:
                spec = output_schema[0]
                base_t = spec["type"]
                shp = _as_shape(spec.get("shape"))
                for pred in js["predictions"]:
                    self.assertTrue(
                        matches_schema(pred, base_t, shp),
                        f"Prediction shape/type mismatch for model '{model_name}': "
                        f"expected type={base_t}, shape={shp}, got={pred}"
                    )

    def test_invalid(self):
        """Send an invalid request (missing required field) and expect 422."""
        model_name, cfg = next(iter(self.model_config_dict.items()))
        input_schema = cfg["schema"]["input"]
        inst = build_instance(input_schema)
        inst.pop(input_schema[0]["name"], None)
        r = self.client.post(f"/models/{model_name}/predict", json={"instances": [inst]})
        self.assertEqual(r.status_code, 422)
        self.assertIn("detail", r.json())