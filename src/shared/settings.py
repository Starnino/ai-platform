from pathlib import Path
import os
import logging
from dotenv import load_dotenv
from functools import lru_cache

logger = logging.getLogger(__name__)

class BaseSettings:

    def __init__(self, env_file: str | None = ".env") -> None:
        """
        Base settings class to manage environment variables and application settings.
        Args:
            env_file (str | None): Path to the .env file. If None, environment variables are not loaded from a file.
        """
        # load environment variables from a file if provided
        if env_file: 
            if os.path.exists(env_file):
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from {Path(env_file)}")
            else:
                logger.warning(f"Environment file {env_file} not found. Skipping.")

        # defaults
        PROJECT_DIR = Path(os.getenv("PROJECT_DIR", None) or self._find_project_root(Path(__file__).resolve().parent))
        if not PROJECT_DIR.exists():
            raise RuntimeError(f"PROJECT_DIR does not exist: {PROJECT_DIR}")
        self.CONFIG_DIR = Path(os.getenv("CONFIG_DIR", PROJECT_DIR / "config"))
        self.DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_DIR / "data"))
    
    def get(self, key: str, default=None, cast: type | None = None):
        """
        Get an environment variable with optional casting and default value.
        Args:
            key (str): Environment variable key.
            default: Default value if the environment variable is not set.
            cast (type | None): Type to cast the environment variable value to.
        Returns:
            The value of the environment variable, cast to the specified type, or the default value.
        """
        value = os.getenv(key)
        if value is None:
            logger.warning(f"Environment variable {key} not found. Using default: {default}")
            value = default
        elif value == "":
            logger.warning(f"Environment variable {key} is empty. Using default: {default}")
            value = default
        else:
            if any(kw in key.lower() for kw in ["psw", "password", "secret", "token", "key"]):
                print_value = "*" * len(str(value))
            else:
                print_value = value
            logger.info(f"Environment variable {key} found with value: {print_value}")

        if cast:
            try:
                return cast(value)
            except Exception as e:
                raise ValueError(
                f"Failed to cast env var {key}={value!r} to {cast.__name__}"
            ) from e

        return value

    # Make the class read-only after initialization
    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(
                f"{self.__class__.__name__} is read-only. Cannot reassign '{name}'"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError(f"{self.__class__.__name__} is read-only")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(DATA_DIR={self.DATA_DIR}, CONFIG_DIR={self.CONFIG_DIR})"
    
    @staticmethod
    def _find_project_root(start: Path, markers=("config", "src", "data")) -> Path:
        """
        Find the project root directory by looking for specific marker directories.
        """
        for p in [start] + list(start.parents):
                if any((p / m).exists() for m in markers):
                    return p
        return start

    # -----------------------------
    # Factory singleton
    # -----------------------------
    @classmethod
    @lru_cache(maxsize=1)
    def get_settings(cls, *, env_file: str | None = None):
        """
        Factory method to get a singleton instance of the settings.
        """
        return cls(env_file=env_file)