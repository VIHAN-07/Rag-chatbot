"""
Pipeline module for orchestrating the complete voice RAG chatbot flow.

This module provides a clean, high-level interface for end-to-end processing,
hiding the complexity of component interactions.

=============================================================================
MODULE RESPONSIBILITY (Facade Pattern)
=============================================================================
This module provides a UNIFIED INTERFACE for:
✓ Combining STT, RAG, LLM, and TTS into a single flow
✓ Simplifying component orchestration
✓ Providing a clean API for external callers

DESIGN PATTERN: Facade
- Hides complexity of subsystem interactions
- Provides simple interface for common use cases
- Delegates work to appropriate components

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why a separate pipeline module?"
   → Separation of concerns - UI doesn't need to know component details

2. "What pattern is this?"
   → Facade pattern - simplifies complex subsystem interactions

3. "How does this support testing?"
   → Each component can be mocked independently, pipeline can be tested
     with mock components

=============================================================================
DATA FLOW
=============================================================================
Voice Input:
  Audio → [STT] → Text → [Retriever] → Context → [LLM] → Response → [TTS] → Audio

Text Input:
  Query → [Retriever] → Context → [LLM] → Response → [TTS] → Audio
"""

from typing import Optional, Tuple, Union, Final
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from app.config import get_config
from app.stt import get_stt, transcribe
from app.retriever import get_retriever
from app.llm import get_llm_handler
from app.tts import get_tts


# =============================================================================
# RESULT DATACLASSES (Clean API Design)
# =============================================================================

@dataclass
class TextProcessingResult:
    """
    Result of processing a text query.
    
    Provides structured output instead of tuple for better readability.
    """
    query: str
    response: str
    audio_bytes: Optional[bytes] = None
    
    @property
    def has_audio(self) -> bool:
        """Check if audio was generated."""
        return self.audio_bytes is not None


@dataclass
class VoiceProcessingResult:
    """
    Result of processing a voice input.
    
    Extends TextProcessingResult with transcription.
    """
    transcribed_text: str
    response: str
    audio_bytes: Optional[bytes] = None
    
    @property
    def has_audio(self) -> bool:
        """Check if audio was generated."""
        return self.audio_bytes is not None


class VoiceRAGPipeline:
    """
    End-to-end pipeline for voice-enabled RAG chatbot.
    
    FACADE PATTERN:
    Provides a simplified interface to the complex subsystem of
    STT, Retrieval, LLM, and TTS components.
    
    ORCHESTRATION FLOW:
    1. STT: Audio → Text (if voice input)
    2. RAG: Query → Retrieved Context
    3. LLM: Context + Query → Response  
    4. TTS: Response → Audio (optional)
    
    USAGE:
        pipeline = VoiceRAGPipeline()
        result = pipeline.process_text("How do I reset my password?")
        print(result.response)
    
    INTERVIEW TALKING POINT:
    "The pipeline encapsulates the orchestration logic, making the
    Streamlit UI code simpler and enabling easier testing."
    """
    
    def __init__(self, use_mock_llm: bool = True):
        """
        Initialize the pipeline components.
        
        LAZY LOADING: Components are loaded via singleton factories,
        so they're only initialized once across multiple pipeline instances.
        
        Args:
            use_mock_llm: Use mock LLM (default True for zero-cost).
        """
        self.config = get_config()
        self.stt = get_stt()
        self.retriever = get_retriever()
        self.llm = get_llm_handler(use_mock=use_mock_llm)
        self.tts = get_tts()
    
    def process_text(
        self, 
        query: str,
        generate_audio: bool = True
    ) -> TextProcessingResult:
        """
        Process a text query through the pipeline.
        
        FLOW: Query → RAG → LLM → Response → [TTS] → Audio
        
        Args:
            query: User's text query.
            generate_audio: Whether to generate audio response.
            
        Returns:
            TextProcessingResult with response and optional audio.
        """
        # DEFENSIVE: Validate input
        query = (query or "").strip()
        if not query:
            return TextProcessingResult(
                query=query,
                response="I didn't receive a question. How can I help you?",
                audio_bytes=None
            )
        
        # Generate text response using RAG
        text_response = self.llm.generate_response(query)
        
        # Generate audio if requested
        audio_bytes = None
        if generate_audio:
            try:
                audio_bytes = self.tts.synthesize_to_bytes(text_response)
            except Exception as e:
                print(f"Warning: TTS failed: {e}")
        
        return TextProcessingResult(
            query=query,
            response=text_response,
            audio_bytes=audio_bytes
        )
    
    def process_voice(
        self, 
        audio_input: Union[str, Path, bytes, np.ndarray],
        generate_audio: bool = True
    ) -> VoiceProcessingResult:
        """
        Process a voice input through the complete pipeline.
        
        FLOW: Audio → STT → Query → RAG → LLM → Response → [TTS] → Audio
        
        Args:
            audio_input: Audio file path, bytes, or numpy array.
            generate_audio: Whether to generate audio response.
            
        Returns:
            VoiceProcessingResult with transcription, response, and optional audio.
        """
        # Step 1: STT - Convert audio to text
        try:
            transcribed_text = transcribe(audio_input)
        except Exception as e:
            return VoiceProcessingResult(
                transcribed_text=f"[Transcription failed: {e}]",
                response="I couldn't understand the audio. Please try again.",
                audio_bytes=None
            )
        
        # Step 2 & 3: RAG + LLM - Generate response
        result = self.process_text(transcribed_text, generate_audio)
        
        return VoiceProcessingResult(
            transcribed_text=transcribed_text,
            response=result.response,
            audio_bytes=result.audio_bytes
        )
    
    def ingest_documents(self, docs_path: Optional[str] = None) -> int:
        """
        Ingest documents into the knowledge base.
        
        DELEGATION: Simply passes through to retriever.
        
        Args:
            docs_path: Path to documents directory.
            
        Returns:
            Number of document chunks ingested.
        """
        return self.retriever.ingest_documents(docs_path)
    
    def is_knowledge_base_ready(self) -> bool:
        """Check if the knowledge base has been populated."""
        return self.retriever.is_ready()
    
    def clear_conversation(self) -> None:
        """Clear the conversation history."""
        self.llm.clear_history()
    
    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.llm.get_history()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pipeline(use_mock_llm: bool = True) -> VoiceRAGPipeline:
    """
    Factory function to create a pipeline instance.
    
    DESIGN: Factory pattern allows flexible instantiation and
    future extension (e.g., different pipeline configurations).
    
    Args:
        use_mock_llm: Use mock LLM (default True for zero-cost).
        
    Returns:
        Configured VoiceRAGPipeline instance.
    """
    return VoiceRAGPipeline(use_mock_llm=use_mock_llm)


# =============================================================================
# EXAMPLE USAGE (for testing/demonstration)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Voice RAG Pipeline Demo")
    print("=" * 60)
    
    # Create pipeline (use mock for demo - zero cost)
    pipeline = create_pipeline(use_mock_llm=True)
    
    # Ingest sample documents
    print("\nIngesting documents...")
    try:
        count = pipeline.ingest_documents()
        print(f"✓ Ingested {count} chunks")
    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
    
    # Test text query
    print("\n" + "-" * 40)
    print("Testing text query...")
    query = "How do I reset my password?"
    result = pipeline.process_text(query, generate_audio=False)
    print(f"Query: {query}")
    print(f"Response: {result.response}")
    print(f"Audio generated: {result.has_audio}")
