import unittest
from fastapi.testclient import TestClient
from apps.serving.gateway.app import app as gateway
from apps.serving.transformer.app import app as transformer
from apps.serving.predictor.app import app as predictor

class TestApp(unittest.TestCase):
    def setUp(self):
        self.gateway_client = TestClient(gateway)
        self.transformer_client = TestClient(transformer)
        self.predictor_client = TestClient(predictor)

    def test_root_endpoint(self):
        response = self.gateway_client.get("/")
        self.assertEqual(response.status_code, 200)
        response = self.transformer_client.get("/")
        self.assertEqual(response.status_code, 200)
        response = self.predictor_client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_health(self):
        response = self.gateway_client.get("/health")
        self.assertEqual(response.status_code, 200)
        response = self.transformer_client.get("/health")
        self.assertEqual(response.status_code, 200)
        response = self.predictor_client.get("/health")
        self.assertEqual(response.status_code, 200)
    
    def test_gateway(self):
        response = self.gateway_client.get("/models")
        self.assertEqual(response.status_code, 200)

    def test_transformer(self):
        response = self.transformer_client.get("/transformers")
        self.assertEqual(response.status_code, 200)

    def test_predictor(self):
        response = self.predictor_client.get("/models")
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()