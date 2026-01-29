"""
Voice RAG Chatbot - Application Module

This package contains the core components of the voice-enabled
RAG chatbot for customer support.

Components:
- stt: Speech-to-Text using Whisper
- retriever: Document retrieval using ChromaDB
- llm: LLM inference for response generation
- tts: Text-to-Speech for audio output
- config: Configuration management
"""

from app.config import get_config, Config
from app.stt import SpeechToText, get_stt, transcribe
from app.retriever import DocumentRetriever, get_retriever
from app.llm import RAGLLMHandler, get_llm_handler
from app.tts import TextToSpeech, get_tts, text_to_audio

__all__ = [
    'Config',
    'get_config',
    'SpeechToText',
    'get_stt',
    'transcribe',
    'DocumentRetriever',
    'get_retriever',
    'RAGLLMHandler',
    'get_llm_handler',
    'TextToSpeech',
    'get_tts',
    'text_to_audio',
]

__version__ = "1.0.0"
