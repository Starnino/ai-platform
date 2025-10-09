from src.shared.settings import BaseSettings

class Settings(BaseSettings):
    def __init__(self, env_file: str | None = ".env") -> None:
        super().__init__()
        
        # Paths
        self.TRANSFORMERS_CONFIG_PATH = self.CONFIG_DIR / "transformers.yaml"

        # Transformer
        self.HOST = self.get("HOST", "0.0.0.0")
        self.PORT: int = self.get("PORT", 5001, int)
        self.TIMEOUT: int = self.get("TIMEOUT", 60, int)
        self.PROTOCOL = self.get("PROTOCOL", "v1")
        self.PRELOAD_MODELS: bool = self.get("PRELOAD_MODELS", False, bool)

        # Predictor
        PREDICTOR_HOST = self.get("PREDICTOR_HOST", "127.0.0.1")
        PREDICTOR_PORT: int = self.get("PREDICTOR_PORT", 5002, int)
        self.PREDICTOR_PROTOCOL = self.get("PREDICTOR_PROTOCOL", "v1")
        self.PREDICTOR_URL = f"http://{PREDICTOR_HOST}:{PREDICTOR_PORT}"