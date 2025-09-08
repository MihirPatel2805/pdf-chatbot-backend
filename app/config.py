"""
Configuration management for the PDF Chatbot Backend.
Handles environment variables and application settings.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = Field(default="PDF Chatbot Backend", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Google AI Configuration
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    google_embedding_model: str = Field(default="models/embedding-001", env="GOOGLE_EMBEDDING_MODEL")
    google_chat_model: str = Field(default="gemini-1.5-flash", env="GOOGLE_CHAT_MODEL")
    google_temperature: float = Field(default=0.1, env="GOOGLE_TEMPERATURE")
    google_max_tokens: int = Field(default=500, env="GOOGLE_MAX_TOKENS")
    
    # Qdrant Configuration
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_prefix: str = Field(default="pdf-chatbot", env="QDRANT_COLLECTION_PREFIX")
    
    # Vector Database Configuration
    vector_dimension: int = Field(default=768, env="VECTOR_DIMENSION")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    similarity_search_k: int = Field(default=3, env="SIMILARITY_SEARCH_K")
    
    # File Processing Configuration
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    allowed_file_types: list = Field(default=["pdf"], env="ALLOWED_FILE_TYPES")
    
    # Security Configuration
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Clerk Authentication Configuration
    clerk_secret_key: str = Field(..., env="CLERK_SECRET_KEY")
    clerk_publishable_key: str = Field(..., env="CLERK_PUBLISHABLE_KEY")
    clerk_jwks_url: str = Field(default="https://wise-porpoise-12.clerk.accounts.dev/.well-known/jwks.json", env="CLERK_JWKS_URL")
    clerk_issuer: str = Field(default="https://wise-porpoise-12.clerk.accounts.dev", env="CLERK_ISSUER")
    clerk_audience: str | None = Field(default=None, env="CLERK_AUDIENCE")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/pdf_chatbot",
        env="DATABASE_URL"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from environment


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def validate_required_settings() -> None:
    """Validate that all required settings are present."""
    required_settings = [
        ("google_api_key", settings.google_api_key),
        ("clerk_secret_key", settings.clerk_secret_key),
        ("clerk_publishable_key", settings.clerk_publishable_key),
    ]
    
    missing_settings = []
    for setting_name, setting_value in required_settings:
        if not setting_value:
            missing_settings.append(setting_name)
    
    if missing_settings:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_settings)}. "
            "Please check your .env file."
        )
