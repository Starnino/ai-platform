from fastapi import FastAPI, HTTPException, Body, Depends
from contextlib import asynccontextmanager
from src.shared.loader import load_yaml
from src.core.models.loader import load_models
from src.core.models.registry import ModelRegistry
import uvicorn
import logging
from src.shared.logging import configure_logging
from .settings import Settings
import numpy as np

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)

# Load settings
cfg = Settings.get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    # Load model configurations
    model_config = load_yaml(cfg.MODELS_CONFIG_PATH)
    models = load_models(model_config, cfg.MODELS_DATA_DIR)
    registry = ModelRegistry(models)
    if cfg.PRELOAD_MODELS:
        registry.preload()
    app.state.registry = registry
    yield
    # --- SHUTDOWN ---

def get_registry() -> ModelRegistry:
    return app.state.registry

# Create app
name = "predictor"
app = FastAPI(title=name, lifespan=lifespan)

@app.get("/")
def root():
    return {"message": f"{name} is running"}

@app.get("/ready", include_in_schema=False)
async def readiness():
    """
    Readiness check endpoint.
    """
    return {"status": "ready"}

@app.get("/healthz", include_in_schema=False)
def liveness():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/models")
def list_models(registry: ModelRegistry = Depends(get_registry)):
    """
    List all available models.
    """
    return {"models": list(registry.keys())}

@app.get("/models/{model_name}")
def summary(model_name: str, registry: ModelRegistry = Depends(get_registry)):
    """
    Summary API to get the summary of the specified model.
    """
    # Check if the model exists
    if model_name not in registry:
        logger.error(f"Model '{model_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.") 
    
    # Get the model
    model = registry[model_name]

    # Return the summary of the model
    return {"summary": model.summary()}

@app.post("/v1/models/{model_name}:predict")
def predict(model_name: str, payload: dict = Body(..., example={'instances': []}), registry: ModelRegistry = Depends(get_registry)):
    """
    Predict API the output using the specified model.
    """ 
    # Check if the model exists
    if model_name not in registry:
        logger.error(f"Model '{model_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.") 
    
    # Get the model and version
    model = registry[model_name] 

    # Get data from the payload
    data = payload.get("instances")

    # Execute the model prediction asynchronously
    try:
        # lazy loading of the model
        if not model.is_loaded:
            logger.info(f"Loading model '{model_name}' for prediction.")
            model.load()
        else:
            logger.info(f"Model '{model_name}' is already loaded.")
        # Run prediction
        X = np.array(data)
        y_pred = model.predict(X)
        return {"predictions": y_pred.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT, timeout_keep_alive=cfg.TIMEOUT, log_config=None)