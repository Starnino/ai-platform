"""
TensorflowModel

A specialized subclass of BaseModel designed for TensorFlow/Keras models. This
class expects the user to implement `_build_impl()` to define the actual Keras
architecture (e.g., a tf.keras.Sequential). Once `build()` is called, the model
is compiled automatically with the optimizer, loss, and metrics provided in the
constructor.

Usage:
------
1. Subclass `TensorflowModel` and implement the `_build_impl()` method, returning 
   an uncompiled tf.keras.Model object.
2. Optionally override `preprocess(data: dict)` for input transformations.
3. Call `fit(X, y)`, `predict(X)`, or `evaluate(X, y)` to train or use the model.
4. Use `save()` and `load()` for versioned persistence.
"""
import tensorflow as tf
import os
import logging
from abc import abstractmethod
from ..base import BaseModel

logger = logging.getLogger(__name__)

class TensorflowModel(BaseModel):
    def __init__(self, name: str, model_path: str, **kwargs):
        """
        Initialize the model instance with all necessary information.

        Parameters
        ----------
        name : str
            Name of the model.
        model_path : str
            Relative path where model checkpoints will be saved/loaded.
        """
        super().__init__(tf.keras.Model, name, model_path, framework='tensorflow', **kwargs)
        self.optimizer = self._parse_optimizer(kwargs['optimizer']) if 'optimizer' in kwargs else None
        self.loss = self._parse_loss(kwargs['loss']) if 'loss' in kwargs else None
        self.metrics = self._parse_metrics(kwargs['metrics']) if 'metrics' in kwargs else []
        self.callbacks = self._parse_callbacks(kwargs['callbacks']) if 'callbacks' in kwargs else []
        self.is_compiled = False
    
    @property
    def model_filename(self):
        return "model.keras"

    @abstractmethod
    def model(self) -> tf.keras.Model:
        """
        Build the model architecture and return an uncompiled Keras model.
        This method must be implemented by subclasses.
        
        Returns
        -------
        tf.keras.Model
            The built Keras model.
        """
        pass

    def build(self):
        self._model = self.model()
    
    def compile(self):
        self._model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
        def _get_config_name(obj):
            if hasattr(obj, "get_config"):
                return obj.get_config().get("name", str(obj))
            elif hasattr(obj, "__name__"):
                return obj.__name__
            else:
                return str(obj)
            
        logger.info(
            f"Model '{self.name}' is being compiled with "
            f"optimizer={_get_config_name(self.optimizer)}, "
            f"loss={_get_config_name(self.loss)}, "
            f"metrics={[ _get_config_name(m) for m in self.metrics ]}."
        )

        self.is_compiled = True

    def fit(self, X, y, **kwargs):
        if self._model is None:
            self.build()
        if not self.is_compiled:
            raise RuntimeError(f"Model '{self.name}' must be compiled before training. Call compile() first.")
        logger.info(f"Training model '{self.name}' on {len(X)} samples.")
        return self._model.fit(X, y, callbacks=self.callbacks, **kwargs)

    def predict(self, X, **kwargs):
        if self._model is None:
            self.build()
        logger.info(f"Predicting with model '{self.name}' on {len(X)} samples.")
        return self._model.predict(X, **kwargs)

    def evaluate(self, X, y, **kwargs):
        if self._model is None:
            self.build()
        if not self.is_compiled:
            raise RuntimeError(f"Model '{self.name}' must be compiled before evaluation. Call compile() first.")
        logger.info(f"Evaluating model '{self.name}' on {len(X)} samples.")
        return self._model.evaluate(X, y, **kwargs)

    def _save(self, path):
        self._model.export(os.path.dirname(path)) # SavedModel format for TensorFlow Serving
        self._model.save(path)

    def _load(self, path):
        self._model = tf.keras.models.load_model(path)

    def summary(self) -> dict:
        """
        Returns (input_shape, output_shape) from the Keras model.
        """
        if self._model is None:
            self.build()

        input_shape = str(tuple(self._model.inputs[0].shape)).replace("None", "-1")
        output_shape = str(tuple(self._model.outputs[0].shape)).replace("None", "-1")

        return {
            "name": self.name,
            "framework": self.framework,
            "version": self.version,
            "description": self.description,          
            "input_shape": input_shape,
            "output_shape": output_shape
        }

    def _parse_optimizer(self, optimizer):
        """Convert optimizer string to a Keras optimizer if necessary."""
        if isinstance(optimizer, str):
            return tf.keras.optimizers.get(optimizer)
        return optimizer

    def _parse_loss(self, loss):
        """Convert loss string to a Keras loss function if necessary."""
        if isinstance(loss, str):
            return tf.keras.losses.get(loss)
        return loss

    def _parse_metrics(self, metrics):
        """Convert metrics strings to Keras metric functions if necessary."""
        return [tf.keras.metrics.get(m) if isinstance(m, str) else m for m in metrics]
    
    def _parse_callbacks(self, callbacks):
        """
        Convert a list of callback configurations into Keras callback objects.
        
        Each callback configuration should be a dictionary with a single key,
        the callback name (as defined in tf.keras.callbacks), and its value is a dictionary
        of parameters to initialize the callback.
        
        Example input:
            [
                {"EarlyStopping": {"monitor": "val_loss", "patience": 10, "verbose": 1}},
                {"ModelCheckpoint": {"filepath": "best_model.h5", "monitor": "val_loss", "save_best_only": True, "verbose": 1}}
            ]
        
        Returns:
        -------
        list
            A list of Keras callback instances.
        """
        parsed_callbacks = []
        if not isinstance(callbacks, list):
            raise ValueError("Callbacks should be provided as a list.")
        for cb in callbacks:
            if not isinstance(cb, dict) or len(cb) != 1:
                raise ValueError("Each callback configuration must be a dictionary with a single key (the callback name).")
            for cb_name, cb_params in cb.items():
                try:
                    callback_class = getattr(tf.keras.callbacks, cb_name)
                except AttributeError:
                    raise ValueError(f"Callback '{cb_name}' not found in tf.keras.callbacks.")
                # If the callback is ModelCheckpoint, modify the filepath parameter
                if cb_name == "ModelCheckpoint":
                    # Ensure the 'filepath' parameter is specified
                    if 'filepath' in cb_params:
                        if not os.path.isabs(cb_params['filepath']):
                            # Append the model folder path to the file path if it's relative
                            cb_params['filepath'] = os.path.join(self.model_path, cb_params['filepath'])
                    else:
                        raise ValueError("ModelCheckpoint callback requires a 'filepath' parameter.")
                try:
                    parsed_callbacks.append(callback_class(**cb_params))
                except TypeError as e:
                    raise ValueError(f"Errore nella configurazione del callback '{cb_name}': {e}")
        return parsed_callbacks