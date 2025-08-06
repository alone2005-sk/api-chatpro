"""
Configuration settings for DAMN BOT AI System
"""
from typing import ClassVar
import os
from typing import List, Dict
from pydantic_settings import BaseSettings

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict


class Settings(BaseSettings):
    USER_AGENT: ClassVar[str] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = True

    CORS_ORIGINS: List[str] = ["*"]

    DATABASE_URL: str = "sqlite:///./data/database.db"

    UPLOAD_DIR: str = "./uploads"
    PROJECT_DATA_DIR: str = "./projects"
    TEMP_DIR: str = "./temp"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024

    # API Keys
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    FIREWORKS_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""
    SERPER_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    REPLICATE_API_KEY: str = ""

    # Local LLM Config
    LOCAL_LLM_ENABLED: bool = False
    LOCAL_LLM_MODEL_PATH: str = "./models"
    LOCAL_LLM_GPU_LAYERS: int = 35

    OLLAMA_HOST: str = "localhost:11434"
    LLAMACPP_HOST: str = "localhost:8080"
    GPT4ALL_HOST: str = "localhost:4891"

    DEFAULT_MODELS: Dict[str, str] = {
        "openai": "gpt-4-turbo-preview",
        "anthropic": "claude-3-opus-20240229",
        "google": "gemini-pro",
        "groq": "mixtral-8x7b-32768",
        "cohere": "command-r-plus",
        "local": "llama-2-70b-chat"
    }

    CODE_EXECUTION_ENABLED: bool = True
    CODE_TIMEOUT: int = 30
    ALLOWED_LANGUAGES: List[str] = [
        "python", "javascript", "typescript", "java", "cpp", "c", "go", "rust", "php", "ruby"
    ]

    VOICE_ENABLED: bool = True
    DEFAULT_VOICE: str = "alloy"
    VOICE_SPEED: float = 1.0

    WEB_SEARCH_ENABLED: bool = True
    MAX_SEARCH_RESULTS: int = 10

    MAX_REQUESTS_PER_MINUTE: int = 60
    ENABLE_RATE_LIMITING: bool = True
    ALLOWED_FILE_TYPES: List[str] = [
        ".txt", ".pdf", ".docx", ".xlsx", ".csv", ".json", ".xml", ".html",
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".php", ".rb",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
        ".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv"
    ]

    DEEP_LEARNING_ENABLED: bool = True
    RESEARCH_MODE_ENABLED: bool = True
    AUTO_CODE_FIX_ENABLED: bool = True
    MULTI_LLM_ORCHESTRATION: bool = True

    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600

    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/damn_bot.log"

    MAX_PROMPT_WORDS: int = 500
    MAX_FILE_SIZE_MB: int = 50

    GROK_LOCAL_MODEL: str = "grok-tiny-llm"
    GROK_LOCAL_URL: str = "http://localhost:11434/v1/chat"

    model_config = SettingsConfigDict(env_file=".env", extra="allow", case_sensitive=True)
    # search ids api key 

    GOOGLE_CSE_ID: ClassVar[str] = "f46716273d0324d8a"
    GOOGLE_API_KEY : ClassVar[str] = "AIzaSyAhtdYfHNYP2IwIE_OMsMtYZJnt0BVs4vY"

# Global settings instance
_settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return _settings
