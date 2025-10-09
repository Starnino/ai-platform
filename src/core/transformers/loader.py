import logging
import importlib
import os
from .helpers import _split_params
from .base import Transformer

logger = logging.getLogger(__name__)

def import_class(module_path: str, class_name: str):
    """
    Import a class from a given moduleW path.
    Parameters
    ----------
    module_path : str
        The path to the module (e.g., 'src.models.my_model').
    class_name : str
        The name of the class to import from the module.
    Returns 
    -------
    model_class : type
        The imported class.
    Raises  
    ------
    ModuleNotFoundError : If the module cannot be found.
    AttributeError : If the class cannot be found in the module.
    """
    try:
        module = importlib.import_module(f"{module_path}")
    except:
        path, file = os.path.split(module_path.replace('.', os.sep))
        raise ModuleNotFoundError(f"Module '{module_path}' not found. Ensure a file named '{file}.py' exists in {path}.")
    # Retrieve the model class
    try:
        model_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'. Make sure the class is defined and named correctly.")
    return model_class

def load_transformers(transformer_config: dict) -> dict:
    """
    This function loads the model configurations from a YAML file. It reads the model definitions,
    including the model name, preprocessors, postprocessors, and framework type.
    Parameters
    ----------
    config: dict
        A dictionary containing the configuration for transformers.
    Returns
    -------
    models: dict
        A dictionary containing preprocessor and postprocessor instances for each model.
    """
    # Load model configurations
    transformers = {}
    
    for transformer_info in transformer_config:
        if "name" not in transformer_info:
            raise ValueError("Transformer configuration must include 'name'.")
        transformer_name = transformer_info["name"]
        transformer_description = transformer_info.get("description", "")
        transformer_context = transformer_info.get("context", {})

        if transformer_context:
            logger.info(f"Context for transformer '{transformer_name}': {transformer_context}")
            try:
                context_name = transformer_context['name']
                context_source = transformer_context['src']
                context_class = import_class(context_source, context_name)
                context = context_class(**transformer_context.get("parameters", {}))
            except Exception as e:
                logger.error(f"Error importing context '{context_name}' from source '{context_source}': {e}")
                raise
        else:
            context = None

        preprocessors = []
        postprocessors = []
       
        for preprocessor in transformer_info.get("preprocessors", []):
            try:
                preprocessor_source = preprocessor["src"]
                preprocessor_name = preprocessor["name"]
                preprocessor_static_params, preprocessor_dynamic_params = _split_params(preprocessor.get("parameters", {}))
                preprocessor_class = import_class(preprocessor_source, preprocessor_name)
                preprocessor_instance = preprocessor_class(**preprocessor_static_params)
                preprocessor_instance.dynamic_params = preprocessor_dynamic_params
                preprocessors.append(preprocessor_instance)
                logger.info(f"Preprocessor '{preprocessor_name}' imported successfully from '{preprocessor_source}' for transformer '{transformer_name}' with static params: {preprocessor_static_params} and dynamic params: {preprocessor_dynamic_params}.")
            except Exception as e:
                logger.error(f"Error importing preprocessor '{preprocessor_name}' from source '{preprocessor_source}': {e}")
                raise

        for postprocessor in transformer_info.get("postprocessors", []):
            try:
                postprocessor_source = postprocessor["src"]
                postprocessor_name = postprocessor["name"]
                postprocessor_static_params, postprocessor_dynamic_params = _split_params(postprocessor.get("parameters", {}))
                postprocessor_class = import_class(postprocessor_source, postprocessor_name)
                postprocessor_instance = postprocessor_class(**postprocessor_static_params)
                postprocessor_instance.dynamic_params = postprocessor_dynamic_params
                postprocessors.append(postprocessor_instance)
                logger.info(f"Postprocessor '{postprocessor_name}' imported successfully from '{postprocessor_source}' for transformer '{transformer_name}' with static params: {postprocessor_static_params} and dynamic params: {postprocessor_dynamic_params}.")
            except Exception as e:
                logger.error(f"Error importing postprocessor '{postprocessor_name}' from source '{postprocessor_source}': {e}")
                raise

        transformers[transformer_name] = Transformer(context=context, preprocessors=preprocessors, postprocessors=postprocessors)

    if not transformers:
        logger.warning("No transformers found in the configuration file.")
    else:
        logger.info("All preprocessors and postprocessors loaded successfully.")

    return transformers