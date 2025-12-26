"""
Core configuration module.
Loads environment variables with type safety using Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Tenant configuration
    tenant_id_default: str = "org-123"
    
    # Database paths - use absolute path from project root
    db_path: str = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "logmind.sqlite")
    faiss_index_path: str = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "faiss.index")
    
    # Ollama configuration
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen2.5:3b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_api_key: Optional[str] = None
    
    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # UI
    ui_port: int = 3000
    
    # Logging
    log_level: str = "INFO"
    
    # Logs folder - absolute path from project root
    logs_folder: str = str(Path(__file__).parent.parent.parent.parent.parent / "Logs")
    
    @property
    def db_path_resolved(self) -> Path:
        """Get absolute path to database."""
        return Path(self.db_path).resolve()
    
    @property
    def faiss_index_path_resolved(self) -> Path:
        """Get absolute path to FAISS index."""
        return Path(self.faiss_index_path).resolve()
    
    @property
    def logs_folder_resolved(self) -> Path:
        """Get absolute path to logs folder."""
        return Path(self.logs_folder).resolve()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience exports
settings = get_settings()
