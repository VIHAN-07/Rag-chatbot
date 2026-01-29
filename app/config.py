"""
Configuration module for the Voice RAG Chatbot.

Centralizes all configuration settings for easy management and
environment-specific overrides.

=============================================================================
MODULE RESPONSIBILITY (Single Responsibility Principle)
=============================================================================
This module ONLY handles:
✓ Configuration values and defaults
✓ Environment variable loading
✓ Directory initialization

This module does NOT handle:
✗ Business logic (→ other modules)
✗ User input (→ streamlit_app.py)

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why centralized config?"
   → Single source of truth, easy to modify, supports env overrides

2. "Why dataclass over dict?"
   → Type hints, IDE autocomplete, immutability option, validation

3. "How would you add per-environment config?"
   → Load different values based on ENV variable, use .env files

=============================================================================
ZERO-COST DESIGN
=============================================================================
All defaults use zero-cost components:
- MockLLM (no API key required)
- Local Whisper (runs on CPU)
- Local HuggingFace embeddings
- Embedded ChromaDB
- gTTS (free service)
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Final


# =============================================================================
# CONSTANTS
# =============================================================================

# Default values (can be overridden via environment variables)
DEFAULT_WHISPER_MODEL: Final[str] = "base"
DEFAULT_EMBEDDING_MODEL: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE: Final[int] = 500
DEFAULT_CHUNK_OVERLAP: Final[int] = 50
DEFAULT_TOP_K: Final[int] = 3


@dataclass
class Config:
    """
    Application configuration settings.
    
    DESIGN PRINCIPLES:
    - All values have sensible defaults
    - Zero-cost by default (no paid APIs)
    - Environment variables can override defaults
    - Directories are auto-created on initialization
    
    USAGE:
        config = get_config()
        print(config.WHISPER_MODEL)  # "base"
    """
    
    # =========================================================================
    # ChromaDB Settings
    # =========================================================================
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "customer_support_docs"
    
    # =========================================================================
    # Embedding Model (Local, FREE)
    # =========================================================================
    EMBEDDING_MODEL: str = DEFAULT_EMBEDDING_MODEL
    
    # =========================================================================
    # LLM Settings
    # NOTE: Defaults use MockLLM (zero-cost). These apply when swapping
    # to a paid provider (OpenAI, Anthropic, etc.)
    # =========================================================================
    LLM_MODEL: str = "gpt-3.5-turbo"  # Only used if swapping to OpenAI
    LLM_TEMPERATURE: float = 0.3      # Lower = more focused responses
    LLM_MAX_TOKENS: int = 512         # Max response length
    
    # =========================================================================
    # Whisper STT Settings (Local, FREE)
    # =========================================================================
    WHISPER_MODEL: str = DEFAULT_WHISPER_MODEL
    
    # =========================================================================
    # TTS Settings
    # =========================================================================
    TTS_RATE: int = 150      # Words per minute
    TTS_VOLUME: float = 1.0  # 0.0 to 1.0
    
    # =========================================================================
    # RAG Settings (Interview Talking Point: Explain chunk size tradeoffs)
    # =========================================================================
    CHUNK_SIZE: int = DEFAULT_CHUNK_SIZE      # Chars per chunk
    CHUNK_OVERLAP: int = DEFAULT_CHUNK_OVERLAP # Overlap prevents info loss
    TOP_K_RESULTS: int = DEFAULT_TOP_K         # Docs to retrieve
    
    # =========================================================================
    # File Paths
    # =========================================================================
    SUPPORT_DOCS_DIR: str = "./data/support_docs"
    AUDIO_OUTPUT_DIR: str = "./audio_output"
    
    # =========================================================================
    # API Keys (OPTIONAL - only for paid LLM providers)
    # =========================================================================
    OPENAI_API_KEY: Optional[str] = None
    
    def __post_init__(self):
        """
        Post-initialization setup.
        
        - Loads API keys from environment
        - Creates required directories
        - Validates configuration values
        """
        # Load optional API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # DEFENSIVE: Create required directories
        for dir_path in [self.CHROMA_PERSIST_DIR, self.SUPPORT_DOCS_DIR, self.AUDIO_OUTPUT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
        # DEFENSIVE: Validate numeric ranges
        self.LLM_TEMPERATURE = max(0.0, min(1.0, self.LLM_TEMPERATURE))
        self.LLM_MAX_TOKENS = max(1, min(4096, self.LLM_MAX_TOKENS))
        self.CHUNK_SIZE = max(100, min(2000, self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = max(0, min(self.CHUNK_SIZE // 2, self.CHUNK_OVERLAP))
        self.TOP_K_RESULTS = max(1, min(10, self.TOP_K_RESULTS))


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config
