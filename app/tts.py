"""
Text-to-Speech (TTS) module for audio response generation.

Converts text responses to audio for voice output, completing the
voice-enabled chatbot experience.

=============================================================================
MODULE RESPONSIBILITY (Single Responsibility Principle)
=============================================================================
This module ONLY handles:
✓ Text-to-speech conversion
✓ Audio file generation
✓ Text preprocessing for speech

This module does NOT handle:
✗ Response generation (→ llm.py)
✗ Speech-to-text (→ stt.py)
✗ User interface (→ streamlit_app.py)

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why gTTS over paid TTS APIs?"
   → Zero-cost, good quality, simple API. For production, consider
     ElevenLabs or Azure TTS for more natural voices.

2. "Why pyttsx3 as fallback?"
   → Works offline, no internet required. Useful for demos without
     network access.

3. "How would you add streaming TTS?"
   → Replace gTTS with streaming provider (ElevenLabs, Play.ht),
     yield audio chunks instead of full file.

=============================================================================
ZERO-COST DESIGN
=============================================================================
- gTTS: Free Google TTS service (requires internet)
- pyttsx3: Offline fallback (no cost, no internet)
"""

import os
import tempfile
from typing import Optional, Final
from pathlib import Path
import hashlib
from datetime import datetime
import re

import pyttsx3
from gtts import gTTS

from app.config import get_config


# =============================================================================
# CONSTANTS (Defensive Design)
# =============================================================================

# Maximum text length for TTS (prevents abuse/memory issues)
MAX_TEXT_LENGTH: Final[int] = 5000

# Minimum text length for TTS (avoid processing empty strings)
MIN_TEXT_LENGTH: Final[int] = 2

# Supported TTS backends
SUPPORTED_BACKENDS: Final[set] = {"gtts", "pyttsx3"}


class TextToSpeech:
    """
    Text-to-Speech converter supporting multiple backends.
    
    BACKENDS:
    - gTTS (default): Google Text-to-Speech - requires internet, better quality
    - pyttsx3: Offline TTS - no internet, lower quality, faster
    
    USAGE:
        tts = TextToSpeech(backend="gtts")
        audio_bytes = tts.synthesize_to_bytes("Hello, how can I help?")
    
    DEFENSIVE DESIGN:
    - Validates text length before processing
    - Falls back to pyttsx3 if gTTS fails
    - Cleans text for better speech output
    """
    
    def __init__(self, backend: str = "gtts"):
        """
        Initialize the TTS engine.
        
        Args:
            backend: TTS backend to use ("pyttsx3" or "gtts").
            
        Raises:
            ValueError: If backend is not supported.
        """
        # DEFENSIVE: Validate backend
        backend = backend.lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported TTS backend: {backend}. Use: {SUPPORTED_BACKENDS}")
        
        self.config = get_config()
        self.backend = backend
        self._engine = None
        
        if backend == "pyttsx3":
            self._init_pyttsx3()
    
    def _init_pyttsx3(self) -> None:
        """Initialize pyttsx3 engine."""
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.config.TTS_RATE)
            self._engine.setProperty('volume', self.config.TTS_VOLUME)
            
            # Try to set a natural-sounding voice
            voices = self._engine.getProperty('voices')
            if voices:
                # Prefer female voice if available (often sounds more natural)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self._engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            print(f"Warning: pyttsx3 initialization failed: {e}")
            self.backend = "gtts"
    
    def _generate_filename(self, text: str) -> str:
        """Generate a unique filename based on text content."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"response_{timestamp}_{text_hash}.mp3"
    
    def synthesize_to_file(
        self, 
        text: str, 
        output_path: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """
        Convert text to speech and save to file.
        
        DEFENSIVE DESIGN:
        - Validates text is non-empty and within length limits
        - Cleans text to remove markdown/URLs
        - Falls back to pyttsx3 if gTTS fails
        
        Args:
            text: Text to convert to speech.
            output_path: Output file path. Auto-generated if not provided.
            language: Language code for gTTS (default: "en").
            
        Returns:
            Path to the generated audio file.
            
        Raises:
            ValueError: If text is empty or too short.
        """
        # DEFENSIVE: Validate input
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = text.strip()
        
        # DEFENSIVE: Check minimum length
        if len(text) < MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short (min {MIN_TEXT_LENGTH} chars)")
        
        # DEFENSIVE: Truncate excessively long text
        if len(text) > MAX_TEXT_LENGTH:
            print(f"Warning: Text truncated from {len(text)} to {MAX_TEXT_LENGTH} chars")
            text = text[:MAX_TEXT_LENGTH]
        
        # Clean text for better speech output
        text = self._clean_text_for_speech(text)
        
        # Determine output path
        if output_path is None:
            filename = self._generate_filename(text)
            output_path = os.path.join(self.config.AUDIO_OUTPUT_DIR, filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if self.backend == "gtts":
            return self._synthesize_gtts(text, output_path, language)
        else:
            return self._synthesize_pyttsx3(text, output_path)
    
    def _synthesize_gtts(self, text: str, output_path: str, language: str) -> str:
        """Synthesize using Google Text-to-Speech."""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(output_path)
            return output_path
        except Exception as e:
            print(f"gTTS failed: {e}, falling back to pyttsx3")
            self._init_pyttsx3()
            return self._synthesize_pyttsx3(text, output_path)
    
    def _synthesize_pyttsx3(self, text: str, output_path: str) -> str:
        """Synthesize using pyttsx3."""
        if self._engine is None:
            self._init_pyttsx3()
        
        if self._engine is None:
            raise RuntimeError("No TTS engine available")
        
        # pyttsx3 saves as wav, convert path if needed
        if output_path.endswith('.mp3'):
            output_path = output_path.replace('.mp3', '.wav')
        
        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()
        
        return output_path
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean text for better speech synthesis.
        
        Args:
            text: Raw text to clean.
            
        Returns:
            Cleaned text optimized for speech.
        """
        # Remove markdown formatting
        text = text.replace('**', '')
        text = text.replace('*', '')
        text = text.replace('`', '')
        text = text.replace('#', '')
        
        # Remove URLs (they sound bad when spoken)
        import re
        text = re.sub(r'https?://\S+', 'link provided', text)
        
        # Replace common abbreviations
        text = text.replace('e.g.', 'for example')
        text = text.replace('i.e.', 'that is')
        text = text.replace('etc.', 'etcetera')
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def synthesize_to_bytes(self, text: str, language: str = "en") -> bytes:
        """
        Convert text to speech and return as bytes.
        
        Args:
            text: Text to convert.
            language: Language code.
            
        Returns:
            Audio data as bytes.
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            output_path = self.synthesize_to_file(text, tmp_path, language)
            with open(output_path, 'rb') as f:
                return f.read()
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Also clean up wav if pyttsx3 was used
            wav_path = tmp_path.replace('.mp3', '.wav')
            if os.path.exists(wav_path):
                os.remove(wav_path)


# Singleton instance
_tts_instance: Optional[TextToSpeech] = None


def get_tts(backend: str = "gtts") -> TextToSpeech:
    """
    Get or create the singleton TTS instance.
    
    Args:
        backend: TTS backend to use.
        
    Returns:
        TextToSpeech instance.
    """
    global _tts_instance
    if _tts_instance is None or _tts_instance.backend != backend:
        _tts_instance = TextToSpeech(backend=backend)
    return _tts_instance


def text_to_audio(text: str, output_path: Optional[str] = None) -> str:
    """
    Convenience function to convert text to audio file.
    
    Args:
        text: Text to convert.
        output_path: Optional output path.
        
    Returns:
        Path to generated audio file.
    """
    tts = get_tts()
    return tts.synthesize_to_file(text, output_path)
