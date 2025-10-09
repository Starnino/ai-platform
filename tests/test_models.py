import unittest
import tempfile
import shutil
import numpy as np

import tensorflow as tf
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.core.models import TensorflowModel, SklearnModel, TorchModel

class DummyTensorflowModel(TensorflowModel):
    def model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

class DummySklearnModel(SklearnModel):
    def model(self):
        return LogisticRegression()

class DummyPytorchModel(TorchModel):
    def model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(4, 1),
            torch.nn.Sigmoid()
        )

    def fit(self, train_loader, **kwargs):
        model = self._model
        model.train()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    def predict(self, X, **kwargs):
        self._model.eval()
        with torch.no_grad():
            return self._model(X.to(self.device)).cpu().numpy()

    def evaluate(self, val_loader, **kwargs):
        self._model.eval()
        loss_fn = torch.nn.BCELoss()
        total_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = self._model(X.to(self.device)).squeeze()
                loss = loss_fn(preds, y.to(self.device))
                total_loss += loss.item()
        return total_loss / len(val_loader)


class TestBaseModelImplementations(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_tensorflow_model(self):
        model = DummyTensorflowModel(
            name="tf_model",
            model_path=self.temp_dir,
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        X = np.random.rand(20, 4)
        y = np.random.randint(0, 2, size=(20,))
        model.build()
        model.compile()
        model.fit(X, y, epochs=1, verbose=0)
        preds = model.predict(X)
        score = model.evaluate(X, y, verbose=0)
        model.save()
        model.load()
        self.assertEqual(preds.shape[0], X.shape[0])
        self.assertTrue(
            isinstance(score, float) or isinstance(score, list),
            f"Expected float or list of floats, got {type(score)}"
        )

    def test_sklearn_model(self):
        model = DummySklearnModel(
            name="sk_model",
            model_path=self.temp_dir,
        )
        X, y = make_classification(n_samples=20, n_features=4, n_informative=2, n_classes=2)
        model.build()
        model.fit(X, y)
        preds = model.predict(X)
        score = model.evaluate(X, y)
        model.save()
        model.load()
        self.assertEqual(len(preds), 20)
        self.assertTrue(
            isinstance(score, float) or isinstance(score, list),
            f"Expected float or list of floats, got {type(score)}"
        )

    def test_pytorch_model(self):
        model = DummyPytorchModel(
            name="torch_model",
            model_path=self.temp_dir,
        )
        X = torch.rand(20, 4)
        y = torch.randint(0, 2, (20,), dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X, y), batch_size=4)
        model.build()
        model.fit(train_loader, epochs=1)
        preds = model.predict(X)
        score = model.evaluate(train_loader)
        model.save()
        model.load()
        self.assertEqual(preds.shape[0], 20)
        self.assertTrue(
            isinstance(score, float) or isinstance(score, list),
            f"Expected float or list of floats, got {type(score)}"
        )