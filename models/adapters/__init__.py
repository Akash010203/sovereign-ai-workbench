"""
models/adapters/__init__.py
"""
from models.adapters.base import ModelProvider, GenerationConfig
from models.adapters.minilm_adapter import CustomMiniLLMAdapter
from models.adapters.openweight_adapter import OllamaModelAdapter

__all__ = ["ModelProvider", "GenerationConfig", "CustomMiniLLMAdapter", "OllamaModelAdapter"]
