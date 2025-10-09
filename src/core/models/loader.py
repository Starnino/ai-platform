import os
import logging
import importlib

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

def load_models(model_config: dict, data_path: str) -> dict:
    """
    This function loads and initializes machine learning models based on the configuration
    defined in the 'models.yaml' file. It dynamically imports the corresponding model classes
    and instantiates each model with the specified parameters.

    Parameters:
    -----------
    config_path: dict
        A dictionary containing the configuration for models.
    data_path: str
        Path to the directory where model resources are stored.
    
    Returns:
    --------
    models: dict
        An dictionary containing the initialized models.
    """
    # Load model configurations
    models = {}

    for model_info in model_config:
        if "name" not in model_info or "src" not in model_info or "framework" not in model_info:
            raise ValueError(f"Model configuration must include 'name', 'src', and 'framework'.")
        model_name = model_info["name"]
        model_source = model_info["src"]
        model_framework = model_info["framework"]
        model_version = model_info.get("version", "latest")
        model_parameters = model_info.get("parameters", {})
        model_training = model_info.get("training", {})

        logger.info(f"Creating model '{model_name}' version '{model_version}' with framework '{model_framework}' from source '{model_source}'.")

        # load the module from src.models.<module_name>
        try:
            model_class = import_class(model_source, model_name)
        except Exception as e:
            logger.error(f"Error importing model class '{model_name}' from source '{model_source}': {e}")
            raise e

        # Define the model file path based on the common data directory
        model_path = os.path.join(data_path, f"{model_name}")
        os.makedirs(model_path, exist_ok=True)
        
        # Instantiate the model with the provided parameters
        try:
            if model_framework in ["tensorflow", "sklearn", "pytorch", "custom"]:
                model_instance = model_class(
                    name=model_name,
                    model_path=model_path,
                    description=model_info.get("description", None),
                    schema=model_info.get("schema", None),
                    version=model_version,
                    **model_parameters, # Unpack model parameters if any
                    **model_training  # Unpack training parameters if any
                )
            else:
                logger.error(f"Unsupported framework type '{model_framework}'.")
                raise ValueError(f"Unsupported framework type '{model_framework}'.")
            
            logger.info(f"Model instance for '{model_name}' created successfully with parameters: {model_parameters}.")
        except Exception as e:
            logger.error(f"Error instantiating model '{model_name}': {e}")
            raise e

        models[model_name] = model_instance

    logger.info("All models initialized successfully.")

    return models