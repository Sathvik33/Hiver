from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/hiver_support"
    
    LLM_PROVIDER: str = "ollama"
    
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct"
    
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.60
    INTENT_CONFIDENCE_THRESHOLD: float = 0.65
    
    SELECTED_BRAND: str = "AmazonHelp"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
