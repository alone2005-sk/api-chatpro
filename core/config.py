"""
Configuration settings for DAMN BOT
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Basic settings
    APP_NAME: str = "DAMN BOT"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./damn_bot.db", env="DATABASE_URL")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    CORS_ORIGINS: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate limiting
    RATE_LIMIT_CALLS: int = Field(default=100, env="RATE_LIMIT_CALLS")
    RATE_LIMIT_PERIOD: int = Field(default=60, env="RATE_LIMIT_PERIOD")
    
    # File processing
    MAX_FILE_SIZE: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    UPLOAD_DIR: str = Field(default="uploads", env="UPLOAD_DIR")
    PROJECT_DATA_DIR: str = Field(default="project_data", env="PROJECT_DATA_DIR")
    TEMP_DIR: str = Field(default="temp", env="TEMP_DIR")
    
    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    TOGETHER_API_KEY: Optional[str] = Field(default=None, env="TOGETHER_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, env="HUGGINGFACE_API_KEY")
    
    # Local LLM settings
    OLLAMA_HOST: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    LLAMACPP_HOST: str = Field(default="http://localhost:8080", env="LLAMACPP_HOST")
    GPT4ALL_HOST: str = Field(default="http://localhost:4891", env="GPT4ALL_HOST")
    
    # Web search
    SERPAPI_API_KEY: Optional[str] = Field(default=None, env="SERPAPI_API_KEY")
    DUCKDUCKGO_ENABLED: bool = Field(default=True, env="DUCKDUCKGO_ENABLED")
    
    # Voice settings
    TTS_ENGINE: str = Field(default="xtts", env="TTS_ENGINE")  # xtts, bark, coqui
    VOICE_MODEL_PATH: str = Field(default="models/voice", env="VOICE_MODEL_PATH")
    
    # Code execution
    CODE_TIMEOUT: int = Field(default=30, env="CODE_TIMEOUT")
    ENABLE_CODE_EXECUTION: bool = Field(default=True, env="ENABLE_CODE_EXECUTION")
    DOCKER_ENABLED: bool = Field(default=False, env="DOCKER_ENABLED")
    
    # Deep learning
    ENABLE_GPU: bool = Field(default=False, env="ENABLE_GPU")
    MODEL_CACHE_DIR: str = Field(default="models", env="MODEL_CACHE_DIR")
    
    # Research settings
    MAX_RESEARCH_DEPTH: int = Field(default=3, env="MAX_RESEARCH_DEPTH")
    MAX_SOURCES: int = Field(default=10, env="MAX_SOURCES")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
