from app.core.config import settings
from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider
from app.llm.groq import GroqProvider
from app.llm.openrouter import OpenRouterProvider

_current_provider: LLMProvider = None

def get_llm_provider() -> LLMProvider:
    global _current_provider
    if _current_provider is not None:
        return _current_provider

    provider_name = settings.LLM_PROVIDER.lower()
    if provider_name == "groq" and settings.GROQ_API_KEY:
        _current_provider = GroqProvider()
    elif provider_name == "openrouter" and settings.OPENROUTER_API_KEY:
        _current_provider = OpenRouterProvider()
    else:
        # Default to local Ollama provider
        _current_provider = OllamaProvider()

    return _current_provider
