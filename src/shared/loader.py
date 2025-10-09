import yaml
from pathlib import Path
from typing import Union, Dict

TYPE_MAPPING = {"int": int, "float": float, "string": str, "boolean": bool}

def load_yaml(filepath: Union[str, Path]) -> Dict:
    """Load YAML config into dictionary."""
    path = Path(filepath)
    with path.open("r") as f:
        content = yaml.safe_load(f)
        if content is None:
            raise ValueError("YAML file is empty")
        return content

def load_models_config(config_path: str) -> dict:
    # Load model configurations
    model_config = load_yaml(config_path)
    models = {}

    for model_info in model_config:
        if "name" not in model_info or "framework" not in model_info:
            raise ValueError("Model configuration must include 'name' and 'framework'.")
        model_name = model_info["name"]
        model_framework = model_info["framework"]
        model_version = model_info.get("version", "latest")
        model_description = model_info.get("description", "")
        model_schema = model_info.get("schema", {})
        model_transformer = model_info.get("transformer", {})
        
        for schema in ["input", "output"]:
            if schema not in model_schema:
                raise ValueError(f"Schema '{schema}' is missing for model '{model_name}'.")
            for b in model_schema[schema]:
                b["type"] = TYPE_MAPPING.get(b["type"], str)
                b["description"] = b.get("description", "")
                b["shape"] = tuple(b.get("shape", [1]))

        models[model_name] = {
            "version": model_version,
            "schema": model_schema,
            "framework": model_framework,
            "description": model_description,
            "transformer": model_transformer
        }

    return models