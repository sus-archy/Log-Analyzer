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
    models_dir: str = str(Path(__file__).parent.parent.parent.parent.parent / "data" / "models")
    
    # Database connection pool
    db_pool_size: int = 5
    db_pool_timeout: float = 30.0
    
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
    
    # CORS settings
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    cors_allow_credentials: bool = True
    
    # Security & Authentication
    auth_enabled: bool = False  # Set to True in production
    secret_key: str = "CHANGE-THIS-SECRET-KEY-IN-PRODUCTION-use-openssl-rand-hex-32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    admin_username: str = "admin"
    admin_password_hash: str = "admin"  # Change this! Use: python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('your-password'))"
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: str = "100/minute"
    rate_limit_chat: str = "10/minute"
    rate_limit_embed: str = "60/minute"  # Allow continuous training
    rate_limit_ingest: str = "20/minute"
    rate_limit_search: str = "30/minute"
    
    @property
    def db_path_resolved(self) -> Path:
        """Get absolute path to database."""
        return Path(self.db_path).resolve()
    
    @property
    def faiss_index_path_resolved(self) -> Path:
        """Get absolute path to FAISS index."""
        return Path(self.faiss_index_path).resolve()
    
    @property
    def models_dir_resolved(self) -> Path:
        """Get absolute path to models directory."""
        return Path(self.models_dir).resolve()
    
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
