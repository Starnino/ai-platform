from contextlib import asynccontextmanager
from typing import Any, Dict, List, Type
from fastapi import FastAPI, HTTPException, Body
import uvicorn
import logging
from src.shared.logging import configure_logging
from src.shared.http import async_call
from src.shared.loader import load_models_config
from src.shared.metrics import BaseMetrics
from .settings import Settings
from pydantic import create_model, BaseModel, ValidationError
import httpx
import asyncio
import time
import contextlib

# Set up logging
configure_logging()
logger = logging.getLogger(__name__)

# Load settings
cfg = Settings.get_settings()

def build_dynamic_pydantic_model(model_name: str, input_schema: List[Dict[str, Any]]) -> Type[BaseModel]:
    from typing import List as TList

    def get_nested_list_type(base_type: type, depth: int):
        t = base_type
        for _ in range(depth):
            t = TList[t]
        return (t, ...)

    fields = {
        block["name"]: get_nested_list_type(block["type"], len(block["shape"]))
        for block in input_schema
    }
    SingleModel = create_model(model_name, **fields)
    BatchModel = create_model(f"Batch{model_name}", instances=(TList[SingleModel], ...))
    return BatchModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    app.state.http_client = httpx.AsyncClient(timeout=cfg.TIMEOUT)
    # Load model configurations
    app.state.model_config = load_models_config(cfg.MODELS_CONFIG_PATH)
    # Initialize dynamic models
    dynamic_models: Dict[str, Type[BaseModel]] = {}
    for name, config in app.state.model_config.items():
        try:
            dynamic_models[name] = build_dynamic_pydantic_model(name, config["schema"]["input"])
        except Exception as e:
            logger.error("Cannot build dynamic model for %s: %s", name, e)
            raise
    app.state.dynamic_models = dynamic_models
    # Backend health cache ----
    app.state.backend_status = {
        "transformer": {"ready": False, "last_change": time.time()},
        "predictor":   {"ready": False, "last_change": time.time()},
    }
    async def poll_backends():
        client: httpx.AsyncClient = app.state.http_client
        t_url = f"{cfg.TRANSFORMER_URL}/ready"
        p_url = f"{cfg.PREDICTOR_URL}/ready"
        while True:
            for name, url in (("transformer", t_url), ("predictor", p_url)):
                try:
                    resp = await client.get(url, timeout=0.5)
                    now_up = resp.status_code == 200
                except Exception:
                    now_up = False
                st = app.state.backend_status[name]
                if st["ready"] != now_up:
                    st["ready"] = now_up
                    st["last_change"] = time.time()
            await asyncio.sleep(cfg.POLL_INTERVAL)

    poller = asyncio.create_task(poll_backends())
    # --- SHUTDOWN ---
    try:
        yield
    finally:
        poller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poller
        await app.state.http_client.aclose()

# Create app
name = "gateway"
app = FastAPI(title=name, lifespan=lifespan)

# Attach metrics
metrics = BaseMetrics.get_metrics()
metrics.attach(app)

@app.get("/")
def root():
    return {"message": f"{name} is running"}

@app.get("/ready", include_in_schema=False)
async def readiness():
    """
    Readiness check endpoint.
    """
    bs = app.state.backend_status
    now = time.time()

    both_down = (not bs["transformer"]["ready"]) and (not bs["predictor"]["ready"])
    longest_down = now - min(bs["transformer"]["last_change"], bs["predictor"]["last_change"])

    if (not both_down) or (both_down and longest_down < cfg.DOWN_GRACE):
        return {
            "status": "ready",
            "transformer": bs["transformer"]["ready"],
            "predictor": bs["predictor"]["ready"]
        }
    raise HTTPException(status_code=503, detail={
        "status": "not_ready",
        "transformer": bs["transformer"]["ready"],
        "predictor": bs["predictor"]["ready"],
        "since": longest_down
    })

@app.get("/healthz", include_in_schema=False)
async def liveness():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/health")
async def health():
    client: httpx.AsyncClient = app.state.http_client
    results = {}
    for name, url in (("transformer", f"{cfg.TRANSFORMER_URL}/ready"),
                      ("predictor",   f"{cfg.PREDICTOR_URL}/ready")):
        try:
            r = await client.get(url, timeout=1.0)
            results[name] = {"ok": r.status_code == 200, "status": r.status_code}
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return results

@app.get("/models")
def list_models():
    """
    List all available models.
    """
    model_config = app.state.model_config
    return {"models": list(model_config.keys())}

@app.get("/models/{model_name}")
async def summary(model_name: str):
    """
    Get information about a specific model.
    """
    # Check if the model exists
    model_config = app.state.model_config
    if model_name not in model_config:
        logger.error(f"Model '{model_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.") 

    def to_map(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
            m: Dict[str, Any] = {}
            for b in blocks:
                m[b["name"]] = {
                    "type":        b["type"].__name__,
                    "shape":       str(b["shape"]),
                    "description": b["description"]
                }
            return m
    
    summary =  {
        "name": model_name,
        "framework": model_config[model_name]["framework"],
        "version": model_config[model_name]["version"],
        "description": model_config[model_name]["description"],
        "input":  to_map(model_config[model_name]["schema"]["input"]),
        "output": to_map(model_config[model_name]["schema"]["output"]) 
    }

    # Retrieve the Transformer parameters
    if transformer := model_config[model_name]["transformer"]:
        client = app.state.http_client
        logger.info(f"Sending request to transformer '{transformer}'")
        response = await async_call(client, f"{cfg.TRANSFORMER_URL}/transformers/{transformer}", "GET")
        transformer_summary = response.get("summary", {})
        parameters = {}
        all_procs = {**transformer_summary.get("preprocessors", {}), **transformer_summary.get("postprocessors", {})}
        # Keep only dynamic parameters
        for _, specs in all_procs.items():
            if specs and specs.get("dynamic"):
                for pname, pspec in specs["dynamic"].items():
                    parameters[pname] = pspec
        if parameters:
            summary["parameters"] = parameters

    return summary

@app.post("/models/{model_name}/predict")
async def predict(model_name: str, payload: dict = Body(..., example={'instances': []})):
    """
    Predict API the output using the specified model.
    """
    # Check if the model exists
    model_config = app.state.model_config
    if model_name not in model_config:
        logger.error(f"Model '{model_name}' not found.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")    

    # Get the model and its input schema
    model = model_config[model_name]

    # Get parameters for preprocessors/postprocessors if any
    parameters = payload.get('parameters', {})

    # Validate the request payload
    DynamicModel = app.state.dynamic_models[model_name]
    try:
        payload = DynamicModel(**payload).model_dump()
    except ValidationError as e:
        msg = e.errors(include_url=False, include_input=False)
        logger.error(msg)
        raise HTTPException(status_code=422, detail=msg)
    
    # transform data
    client = app.state.http_client
    if transformer := model["transformer"]:
        payload['model_name'] = model_name
        payload['parameters'] = parameters
        logger.info(f"Sending request to transformers '{transformer}'")
        response = await async_call(client, f"{cfg.TRANSFORMER_URL}/{cfg.TRANSFORMER_PROTOCOL}/transformers/{transformer}:transform", "POST", payload)
    else:
        # send the request to the predictor
        logger.info(f"Sending request to predictor '{model_name}'")
        response = await async_call(client, f"{cfg.PREDICTOR_URL}/{cfg.PREDICTOR_PROTOCOL}/models/{model_name}:predict", "POST", payload)

    return response

if __name__ == "__main__":
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT, timeout_keep_alive=cfg.TIMEOUT, log_config=None)