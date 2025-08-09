"""
Configuration settings for DAMN BOT AI System
"""

import os
from typing import List, Dict, Any, Optional, ClassVar
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ───── Server Configuration ─────
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # ───── CORS ─────
    CORS_ORIGINS: List[str] = ["*"]

    # ───── Database & Cache ─────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/ai_agent.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600  # seconds

    # ───── File Handling ─────
    UPLOAD_DIR: str = "./uploads"
    GENERATED_MEDIA_DIR: str = "./generated_media"
    PROJECT_DATA_DIR: str = "./projects"
    TEMP_DIR: str = "./temp"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [
        ".txt", ".pdf", ".docx", ".xlsx", ".csv", ".json", ".xml", ".html",
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs", ".php", ".rb",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
        ".mp3", ".wav", ".mp4", ".avi", ".mov", ".mkv"
    ]

    # ───── API Keys ─────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", "")
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
    CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")

    # ───── User Agent / Static ─────
    USER_AGENT: ClassVar[str] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )

    # ───── LLM Configuration ─────
    DEFAULT_MODEL: str = "gpt-3.5-turbo"
    DEFAULT_MODELS: Dict[str, str] = {
        "openai": "gpt-4-turbo-preview",
        "anthropic": "claude-3-opus-20240229",
        "google": "gemini-pro",
        "groq": "mixtral-8x7b-32768",
        "cohere": "command-r-plus",
        "local": "llama-2-70b-chat"
    }
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.7
    MAX_CHAT_HISTORY: int = 50

    # ───── Local LLM ─────
    LOCAL_LLM_ENABLED: bool = True
    LOCAL_LLM_MODEL_PATH: str = "./models"
    LOCAL_LLM_GPU_LAYERS: int = 35
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLAMACPP_BASE_URL: str = "https://localhost:8080"
    GPT4ALL_HOST: str = "https://localhost:4891"

    # ───── Local Image Generation ─────
    AUTOMATIC1111_URL: str = "http://localhost:7860"
    COMFYUI_URL: str = "http://localhost:8188"

    # ───── Voice ─────
    VOICE_ENABLED: bool = True
    DEFAULT_VOICE: str = "alloy"
    VOICE_SPEED: float = 1.0

    # ───── Web Search ─────
    WEB_SEARCH_ENABLED: bool = True
    MAX_SEARCH_RESULTS: int = 10
    GOOGLE_CSE_ID: ClassVar[str] = "f46716273d0324d8a"

    # ───── Security / Auth ─────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 60
    ENABLE_RATE_LIMITING: bool = True
    MAX_REQUESTS_PER_MINUTE: int = 60

    # ───── Features / Modules ─────
    LLM_CACHE_TTL: int = 3600
    ENABLE_SELF_LEARNING: bool = True
    ENABLE_DEEP_RESEARCH: bool = True
    ENABLE_CODE_EXECUTION: bool = True
    ENABLE_PROJECT_GENERATION: bool = True
    ENABLE_MEDIA_GENERATION: bool = True
    DEEP_LEARNING_ENABLED: bool = True
    RESEARCH_MODE_ENABLED: bool = True
    AUTO_CODE_FIX_ENABLED: bool = True
    MULTI_LLM_ORCHESTRATION: bool = True

    # ───── Background Tasks / Async ─────
    LEARNING_BATCH_SIZE: int = 10
    LEARNING_INTERVAL: int = 30
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300

    # ───── Code Execution ─────
    CODE_EXECUTION_ENABLED: bool = True
    CODE_TIMEOUT: int = 30
    ALLOWED_LANGUAGES: List[str] = [
        "python", "javascript", "typescript", "java", "cpp", "c", "go", "rust", "php", "ruby"
    ]

    # ───── Logging ─────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/damn_bot.log"

    # ───── Prompting & Limits ─────
    MAX_PROMPT_WORDS: int = 500
    MAX_FILE_SIZE_MB: int = 50

    # ───── Custom Local Models ─────
    GROK_LOCAL_MODEL: str = "grok-tiny-llm"
    GROK_LOCAL_URL: str = "http://localhost:11434/v1/chat"

    # ───── Model Config ─────
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow",
        case_sensitive=True
    )


# Global instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
