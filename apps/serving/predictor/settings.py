from src.shared.settings import BaseSettings

class Settings(BaseSettings):
    def __init__(self, env_file: str | None = ".env") -> None:
        super().__init__()
        
        # Paths
        self.MODELS_CONFIG_PATH = self.CONFIG_DIR / "models.yaml"
        self.MODELS_DATA_DIR = self.DATA_DIR / "models"
        
        # Predictor
        self.HOST = self.get("HOST", "0.0.0.0")
        self.PORT: int = self.get("PORT", 5002, int)
        self.TIMEOUT: int = self.get("TIMEOUT", 60, int)
        self.PROTOCOL = self.get("PROTOCOL", "v1")
        self.PRELOAD_MODELS: bool = self.get("PRELOAD_MODELS", False, bool)