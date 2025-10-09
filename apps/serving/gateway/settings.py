from src.shared.settings import BaseSettings

class Settings(BaseSettings):
    def __init__(self, env_file: str | None = ".env") -> None:
        super().__init__()
        
        # Paths
        self.MODELS_CONFIG_PATH = self.CONFIG_DIR / "models.yaml"
        
        # Gateway
        self.HOST = self.get("HOST", "0.0.0.0")
        self.PORT: int = self.get("PORT", 8080, int)
        self.TIMEOUT: int = self.get("TIMEOUT", 60, int)

        # Transformer
        TRANSFORMER_HOST = self.get("TRANSFORMER_HOST", "127.0.0.1")
        TRANSFORMER_PORT: int = self.get("TRANSFORMER_PORT", 5001, int)
        self.TRANSFORMER_PROTOCOL = self.get("TRANSFORMER_PROTOCOL", "v1")
        self.TRANSFORMER_URL = f"http://{TRANSFORMER_HOST}:{TRANSFORMER_PORT}"

        # Predictor
        PREDICTOR_HOST = self.get("PREDICTOR_HOST", "127.0.0.1")
        PREDICTOR_PORT: int = self.get("PREDICTOR_PORT", 5002, int)
        self.PREDICTOR_PROTOCOL = self.get("PREDICTOR_PROTOCOL", "v1")
        self.PREDICTOR_URL = f"http://{PREDICTOR_HOST}:{PREDICTOR_PORT}"

        # Metrics
        self.POLL_INTERVAL = self.get("POLL_INTERVAL", 5.0, float)
        self.DOWN_GRACE = self.get("DOWN_GRACE", 15.0, float)