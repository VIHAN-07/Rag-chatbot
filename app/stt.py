"""
Speech-to-Text (STT) module using OpenAI Whisper.

This module handles the conversion of audio input to text, enabling
voice-based interaction with the chatbot.

=============================================================================
MODULE RESPONSIBILITY (Single Responsibility Principle)
=============================================================================
This module ONLY handles:
✓ Audio file loading and validation
✓ Speech-to-text transcription
✓ Audio format conversion

This module does NOT handle:
✗ Text-to-speech (→ tts.py)
✗ Response generation (→ llm.py)
✗ User interface (→ streamlit_app.py)

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why Whisper over cloud STT?"
   → Free, runs locally, no API costs. For production, consider
     Deepgram or AssemblyAI for streaming and better latency.

2. "What's the latency tradeoff?"
   → Whisper is batch-only. For real-time, use streaming STT.
     This design prioritizes accuracy over latency.

3. "Why fp16=False?"
   → Ensures CPU compatibility. GPU users can enable for speed.

4. "How would you add streaming?"
   → Replace with Deepgram/AssemblyAI WebSocket API, yield
     partial transcripts as audio streams in.

=============================================================================
MODEL SIZE OPTIONS
=============================================================================
| Model  | Size   | Speed  | Accuracy | RAM Needed |
|--------|--------|--------|----------|------------|
| tiny   | 39 MB  | Fast   | Basic    | ~1 GB      |
| base   | 74 MB  | Fast   | Good     | ~1 GB      | ← Default
| small  | 244 MB | Medium | Better   | ~2 GB      |
| medium | 769 MB | Slow   | Great    | ~5 GB      |
| large  | 1550MB | Slow   | Best     | ~10 GB     |

=============================================================================
ZERO-COST DESIGN
=============================================================================
Whisper runs 100% locally after initial model download.
No API calls, no usage limits, no costs.
"""

import whisper
import numpy as np
import tempfile
import os
from typing import Union, Optional, Final
from pathlib import Path

from app.config import get_config


# =============================================================================
# CUSTOM EXCEPTIONS FOR CLEAR ERROR HANDLING
# =============================================================================

class STTError(Exception):
    """
    Base exception for Speech-to-Text errors.
    
    DESIGN: Exception hierarchy allows callers to catch specific
    errors or broad STTError for general handling.
    """
    pass


class AudioFileNotFoundError(STTError):
    """Raised when the audio file does not exist."""
    pass


class InvalidAudioFormatError(STTError):
    """Raised when the audio format is not supported."""
    pass


class TranscriptionFailedError(STTError):
    """Raised when Whisper fails to transcribe the audio."""
    pass


class ModelLoadError(STTError):
    """Raised when the Whisper model fails to load."""
    pass


# =============================================================================
# CONSTANTS (Defensive Design)
# =============================================================================

# Supported audio file extensions
SUPPORTED_FORMATS: Final[set] = {
    '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', 
    '.wma', '.aac', '.mp4', '.mpeg', '.mpga'
}

# Valid Whisper model names
VALID_MODELS: Final[set] = {'tiny', 'base', 'small', 'medium', 'large'}

# Maximum audio file size (100 MB) to prevent memory issues
MAX_FILE_SIZE_BYTES: Final[int] = 100 * 1024 * 1024

# Message returned when no speech is detected
NO_SPEECH_MESSAGE: Final[str] = "[No speech detected in the audio]"


# =============================================================================
# SPEECH-TO-TEXT CLASS
# =============================================================================

class SpeechToText:
    """
    Speech-to-Text converter using OpenAI Whisper.
    
    This class provides methods to transcribe:
    - Audio files (.wav, .mp3, .m4a, etc.)
    - Audio bytes (from uploaded files)
    - Numpy arrays (from audio processing pipelines)
    
    USAGE:
        stt = SpeechToText(model_name="base")
        text = stt.transcribe_file("recording.wav")
        print(text)  # "Hello, I have a question about returns..."
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large).
                       Defaults to config setting ("base").
                       
        Raises:
            ModelLoadError: If the model fails to load.
        """
        config = get_config()
        self.model_name = model_name or config.WHISPER_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the Whisper model into memory.
        
        The model is downloaded automatically on first use (~74 MB for base).
        Subsequent loads use the cached model from disk.
        
        Raises:
            ModelLoadError: If model loading fails.
        """
        print(f"Loading Whisper model: {self.model_name}...")
        try:
            self.model = whisper.load_model(self.model_name)
            print(f"✓ Whisper model '{self.model_name}' loaded successfully.")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model '{self.model_name}': {e}\n"
                "Ensure you have enough memory and the model name is valid.\n"
                "Valid models: tiny, base, small, medium, large"
            )
    
    def transcribe_file(self, audio_path: Union[str, Path]) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Transcribed text from the audio.
            
        Raises:
            AudioFileNotFoundError: If the audio file doesn't exist.
            InvalidAudioFormatError: If the format is not supported.
            TranscriptionFailedError: If transcription fails.
        """
        audio_path = Path(audio_path)
        
        # Validate file exists
        if not audio_path.exists():
            raise AudioFileNotFoundError(
                f"Audio file not found: {audio_path}\n"
                "Please check the file path and try again."
            )
        
        # Validate file format
        suffix = audio_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise InvalidAudioFormatError(
                f"Unsupported audio format: {suffix}\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        
        # Check file is not empty
        if audio_path.stat().st_size == 0:
            raise InvalidAudioFormatError(
                "Audio file is empty. Please provide a valid audio file."
            )
        
        try:
            # Transcribe the audio
            # fp16=False ensures compatibility with CPU-only systems
            result = self.model.transcribe(str(audio_path), fp16=False)
            transcribed_text = result["text"].strip()
            
            # Handle empty transcription
            if not transcribed_text:
                return NO_SPEECH_MESSAGE
            
            return transcribed_text
            
        except Exception as e:
            raise TranscriptionFailedError(
                f"Transcription failed: {str(e)}\n"
                "This could be due to:\n"
                "- Corrupted audio file\n"
                "- Audio too short or silent\n"
                "- FFmpeg not installed (required for audio processing)"
            )
    
    def transcribe_audio_data(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio data (numpy array) to text.
        
        Args:
            audio_data: Audio data as numpy array. Should be 16000 Hz sample rate.
            
        Returns:
            Transcribed text from the audio.
            
        Raises:
            TranscriptionFailedError: If transcription fails.
        """
        # Validate input
        if audio_data is None or len(audio_data) == 0:
            return "[No audio data provided]"
        
        # Ensure audio is float32 and normalized
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if values exceed [-1, 1] range
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        try:
            result = self.model.transcribe(audio_data, fp16=False)
            transcribed_text = result["text"].strip()
            
            if not transcribed_text:
                return NO_SPEECH_MESSAGE
            
            return transcribed_text
            
        except Exception as e:
            raise TranscriptionFailedError(
                f"Transcription of audio data failed: {str(e)}"
            )
    
    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """
        Transcribe audio from bytes data (e.g., from file upload).
        
        This method:
        1. Writes bytes to a temporary file
        2. Transcribes the temporary file
        3. Cleans up the temporary file
        
        Args:
            audio_bytes: Raw audio bytes from file upload.
            suffix: File extension for the temporary file.
            
        Returns:
            Transcribed text from the audio.
            
        Raises:
            InvalidAudioFormatError: If audio bytes are empty.
            TranscriptionFailedError: If transcription fails.
        """
        # Validate input
        if not audio_bytes:
            raise InvalidAudioFormatError(
                "Audio data is empty. Please provide a valid audio file."
            )
        
        # Validate suffix
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        
        # Write bytes to temporary file for Whisper processing
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            result = self.transcribe_file(tmp_path)
            return result
            
        finally:
            # Always clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass  # Ignore cleanup errors


# =============================================================================
# SINGLETON PATTERN FOR EFFICIENCY
# =============================================================================

_stt_instance: Optional[SpeechToText] = None


def get_stt() -> SpeechToText:
    """
    Get or create the singleton STT instance.
    
    Using a singleton ensures the Whisper model is only loaded once,
    which saves memory and speeds up subsequent transcriptions.
    
    Returns:
        The shared SpeechToText instance.
        
    Raises:
        ModelLoadError: If the model fails to load.
    """
    global _stt_instance
    if _stt_instance is None:
        _stt_instance = SpeechToText()
    return _stt_instance


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def transcribe(audio_input: Union[str, Path, bytes, np.ndarray]) -> str:
    """
    Convenience function to transcribe audio from various input types.
    
    Automatically detects input type and calls appropriate method:
    - String/Path: Treats as file path
    - bytes: Treats as raw audio data
    - numpy array: Treats as audio samples
    
    Args:
        audio_input: Audio file path, bytes, or numpy array.
        
    Returns:
        Transcribed text.
        
    Raises:
        STTError: Various subclasses for specific error cases.
        TypeError: If input type is not supported.
    """
    stt = get_stt()
    
    if isinstance(audio_input, (str, Path)):
        return stt.transcribe_file(audio_input)
    elif isinstance(audio_input, bytes):
        return stt.transcribe_bytes(audio_input)
    elif isinstance(audio_input, np.ndarray):
        return stt.transcribe_audio_data(audio_input)
    else:
        raise TypeError(
            f"Unsupported audio input type: {type(audio_input)}\n"
            "Expected: file path (str/Path), bytes, or numpy array."
        )