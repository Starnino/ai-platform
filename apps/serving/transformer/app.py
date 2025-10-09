from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Depends
from src.core.transformers.registry import TransformerRegistry
from src.shared.loader import load_yaml
from src.core.transformers.loader import load_transformers
from src.core.transformers.base import BaseProcessor
import uvicorn
import logging
from src.shared.logging import configure_logging
import httpx
from src.shared.http import async_call
import numpy as np
from .settings import Settings
from typing import Any, Dict, List

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)

# Load settings
cfg = Settings.get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    transformer_config = load_yaml(cfg.TRANSFORMERS_CONFIG_PATH)
    transformers = load_transformers(transformer_config)
    # Initialize the transformer registry
    app.state.registry = TransformerRegistry(transformers)
    app.state.http_client = httpx.AsyncClient(timeout=cfg.TIMEOUT)
    yield
    # --- SHUTDOWN ---
    await app.state.http_client.aclose()

def get_registry() -> TransformerRegistry:
    return app.state.registry
    
def route_params(params: dict, preprocessors: List[BaseProcessor], postprocessors: List[BaseProcessor]) -> dict:
    """
    Route flat parameters to their respective processor based on dynamic_params.
    """
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="parameters must be a dictionary")

    # key -> [(phase, processor)]
    index: Dict[str, List[tuple[str, BaseProcessor]]] = {}

    for p in preprocessors:
        for k in (p.dynamic_params or {}).keys():
            index.setdefault(k, []).append(("pre", p))

    for p in postprocessors:
        for k in (p.dynamic_params or {}).keys():
            index.setdefault(k, []).append(("post", p))

    pre_map: Dict[str, Dict[str, Any]] = {}
    post_map: Dict[str, Dict[str, Any]] = {}

    for k, v in params.items():
        owners = index.get(k, [])
        if not owners:
            raise HTTPException(status_code=400, detail=f"Unknown parameter '{k}' (no processor expects it)")
        if len(owners) > 1:
            clashing = sorted(p.name for p, _ in owners)
            clashing = [f"{phase}:{owner.name}" for phase, owner in owners]
            raise HTTPException(status_code=400, detail=f"Ambiguous parameter '{k}' used by multiple processors: {sorted(clashing)}")
        
        phase, owner = owners[0]
        target = pre_map if phase == "pre" else post_map
        target.setdefault(owner.name, {})[k] = v

    return pre_map, post_map
    
# Create app
name = "transformer"
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

@app.get("/health")
async def health():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/transformers")
def list_transformers(registry: TransformerRegistry = Depends(get_registry)):
    """
    List all available transformers.
    """
    # Return the list of transformer names
    return {"transformers": list(registry.keys())}

@app.get("/transformers/{transformer_name}")
def summary(transformer_name: str, registry: TransformerRegistry = Depends(get_registry)):
    """
    Summary API to get the summary of the specified transformer.
    """
    # Check if the transformer exists
    if transformer_name not in registry:
        logger.error(f"Transformer '{transformer_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Transformer '{transformer_name}' not found.") 
    
    # Get the transformer
    transformer = registry[transformer_name]

    # Return the summary of the transformer
    return {"summary": transformer.summary()}

@app.post("/v1/transformers/{transformer_name}:transform")
async def transform(transformer_name: str, payload: dict = Body(..., example={'instances': []}), registry: TransformerRegistry = Depends(get_registry)):
    """
    Predict API the output using the specified model.
    """
    # Check if the model exists
    if transformer_name not in registry:
        logger.error(f"Transformer '{transformer_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Transformer '{transformer_name}' not found.") 
    
    # Get the transformer and model name
    transformer = registry[transformer_name]
    model_name = payload.pop('model_name', None)
    if not model_name:
        logger.error("Model name is required in the payload.")
        raise HTTPException(status_code=400, detail="Model name is required in the payload.")

    # Get parameters for preprocessors/postprocessors
    params = payload.get("parameters", {})
    if not isinstance(params, dict):
        logger.error("Parameters must be a dictionary.")
        raise HTTPException(status_code=400, detail="Parameters must be a dictionary.") 
    else:
        preprocessor_params, postprocessor_params = route_params(params,  transformer.preprocessors, transformer.postprocessors)

    # Apply preprocessors
    if transformer.preprocessors:
        try:
            data = payload.get("instances")
            X = transformer.preprocess(data, params_map=preprocessor_params)
            payload = {"instances": X.tolist()} if isinstance(X, np.ndarray) else X
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    
    # send the request to the predictor
    logger.info(f"Sending request to predictor '{model_name}'")
    client = app.state.http_client
    response = await async_call(client, f"{cfg.PREDICTOR_URL}/{cfg.PREDICTOR_PROTOCOL}/models/{model_name}:predict", "POST", payload)

    # Apply postprocessors
    if transformer.postprocessors:
        try:
            data = response.get("predictions")
            X = transformer.postprocess(data, params_map=postprocessor_params)
            if isinstance(X, dict):
                return X
            elif isinstance(X, list):
                response = {"predictions": X}
            elif isinstance(X, np.ndarray):
                response = {"predictions": X.tolist()}
            else:
                logger.error("Postprocessor returned an unexpected type.")
                raise HTTPException(status_code=500, detail="Postprocessor returned an unexpected type.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error during postprocessing: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    # Return the response
    return response

if __name__ == "__main__":
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT, timeout_keep_alive=cfg.TIMEOUT, log_config=None)