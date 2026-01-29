"""
Streamlit UI for Voice-Enabled RAG Chatbot.

This module provides the web interface for the customer support chatbot,
supporting both text and voice interaction.

=============================================================================
MODULE RESPONSIBILITY (Single Responsibility Principle)
=============================================================================
This module ONLY handles:
✓ User interface rendering
✓ User input collection (text/voice)
✓ Response display and audio playback
✓ Session state management

This module does NOT handle:
✗ Speech-to-text (→ stt.py)
✗ Document retrieval (→ retriever.py)
✗ Response generation (→ llm.py)
✗ Text-to-speech (→ tts.py)

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why Streamlit over Flask/FastAPI?"
   → Rapid prototyping, built-in state management, easy deployment

2. "How does session state work?"
   → st.session_state persists across reruns within browser session

3. "How would you scale this?"
   → Extract backend to FastAPI, use Redis for sessions, containerize

=============================================================================
ZERO-COST DESIGN
=============================================================================
This application runs with ZERO external API costs:
- MockLLM generates responses from retrieved context
- Whisper runs locally for speech-to-text
- gTTS uses free Google Text-to-Speech service
- ChromaDB runs embedded (no cloud charges)

=============================================================================
DATA FLOW
=============================================================================
User Input → [STT if voice] → Query → Retriever → Context → LLM → Response → [TTS] → Display
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Final

try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_RECORDER_AVAILABLE = True
except ImportError:
    AUDIO_RECORDER_AVAILABLE = False

# Import application modules
from app.config import get_config
from app.stt import get_stt, STTError
from app.retriever import get_retriever, RetrieverError
from app.llm import get_llm_handler
from app.tts import get_tts


# =============================================================================
# CONSTANTS
# =============================================================================

APP_TITLE: Final[str] = "🎙️ Voice RAG Customer Support"
APP_VERSION: Final[str] = "3.0"


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Voice RAG Customer Support",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .status-box {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .status-processing {
        background-color: #FFF3E0;
        border-left: 4px solid #FF9800;
    }
    .status-success {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    .status-error {
        background-color: #FFEBEE;
        border-left: 4px solid #F44336;
    }
    .zero-cost-badge {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """
    Initialize Streamlit session state variables.
    
    Session state persists across reruns within the same browser session,
    allowing us to maintain conversation history and loaded components.
    """
    # Conversation messages list
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Component instances (loaded once, reused)
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = None
    if 'stt' not in st.session_state:
        st.session_state.stt = None
    if 'tts' not in st.session_state:
        st.session_state.tts = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    # User preferences
    if 'enable_voice_response' not in st.session_state:
        st.session_state.enable_voice_response = True
    
    # Status tracking
    if 'components_loaded' not in st.session_state:
        st.session_state.components_loaded = False


# =============================================================================
# COMPONENT LOADING
# =============================================================================

def load_components():
    """
    Load and cache the application components.
    
    Components are loaded once and stored in session state:
    - Retriever: Document retrieval from ChromaDB
    - LLM Handler: Response generation (MockLLM)
    - STT: Speech-to-Text (Whisper)
    - TTS: Text-to-Speech (gTTS)
    """
    if st.session_state.components_loaded:
        return True
    
    status_container = st.empty()
    
    try:
        # Load retriever (ChromaDB + embeddings)
        with status_container.container():
            st.info("🔄 Loading embedding model and knowledge base...")
        if st.session_state.retriever is None:
            st.session_state.retriever = get_retriever()
        
        # Load LLM handler (MockLLM - zero cost)
        with status_container.container():
            st.info("🔄 Initializing response generator...")
        if st.session_state.llm_handler is None:
            st.session_state.llm_handler = get_llm_handler()
        
        # Load TTS (gTTS)
        with status_container.container():
            st.info("🔄 Loading text-to-speech engine...")
        if st.session_state.tts is None:
            st.session_state.tts = get_tts()
        
        # Load STT (Whisper) - may take time for first load
        with status_container.container():
            st.info("🔄 Loading Whisper speech recognition model...")
        if st.session_state.stt is None:
            try:
                st.session_state.stt = get_stt()
            except Exception as e:
                st.warning(f"⚠️ Voice input disabled: {e}")
        
        status_container.empty()
        st.session_state.components_loaded = True
        return True
        
    except Exception as e:
        status_container.empty()
        st.error(f"❌ Failed to load components: {e}")
        return False


# =============================================================================
# AUDIO PROCESSING FUNCTIONS
# =============================================================================

def transcribe_audio(audio_file) -> tuple:
    """
    Transcribe uploaded audio file to text.
    
    Args:
        audio_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (transcribed_text, error_message)
    """
    if st.session_state.stt is None:
        return "", "Speech-to-Text is not available. Please check if Whisper loaded correctly."
    
    # Save uploaded file temporarily
    tmp_path = None
    try:
        # Determine file suffix from uploaded file name
        suffix = Path(audio_file.name).suffix or ".wav"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        # Transcribe the file
        text = st.session_state.stt.transcribe_file(tmp_path)
        return text, None
        
    except STTError as e:
        return "", str(e)
    except Exception as e:
        return "", f"Unexpected error during transcription: {e}"
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def generate_audio_response(text: str) -> Optional[bytes]:
    """
    Generate audio from text response.
    
    Args:
        text: Response text to convert to speech
        
    Returns:
        Audio bytes or None if TTS fails
    """
    if st.session_state.tts is None:
        return None
    
    if not text or len(text.strip()) < 2:
        return None
    
    try:
        audio_bytes = st.session_state.tts.synthesize_to_bytes(text)
        return audio_bytes
    except Exception as e:
        # Log but don't show error - voice is optional
        print(f"TTS warning: {e}")
        return None


# =============================================================================
# QUERY PROCESSING
# =============================================================================

def process_query(query: str, status_placeholder) -> dict:
    """
    Process a user query through the RAG pipeline.
    
    Pipeline stages:
    1. Retrieve relevant context from knowledge base
    2. Generate response using MockLLM with context
    3. Optionally generate audio response
    
    Args:
        query: User's question
        status_placeholder: Streamlit placeholder for status updates
        
    Returns:
        Dictionary with 'text' and optionally 'audio' keys
    """
    if not query or not query.strip():
        return {"text": "Please enter a valid question."}
    
    # Stage 1: Retrieving context
    with status_placeholder.container():
        st.info("🔍 Searching knowledge base for relevant information...")
    
    # Stage 2: Generating response
    with status_placeholder.container():
        st.info("💭 Generating response based on documentation...")
    
    try:
        response_text = st.session_state.llm_handler.generate_response(query)
    except Exception as e:
        response_text = f"I encountered an error while processing your question: {e}"
    
    result = {"text": response_text}
    
    # Stage 3: Generating audio (if enabled)
    if st.session_state.enable_voice_response:
        with status_placeholder.container():
            st.info("🔊 Generating voice response...")
        audio_bytes = generate_audio_response(response_text)
        if audio_bytes:
            result["audio"] = audio_bytes
    
    status_placeholder.empty()
    return result


# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_chat_message(role: str, content: str, audio_bytes: bytes = None):
    """Display a chat message with optional audio playback."""
    with st.chat_message(role):
        st.markdown(content)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")


def sidebar_settings():
    """Render sidebar with settings, controls, and system info."""
    with st.sidebar:
        # Zero-cost badge
        st.markdown('<span class="zero-cost-badge">💚 ZERO API COST</span>', unsafe_allow_html=True)
        st.caption("This app runs entirely offline with no paid API calls.")
        
        st.divider()
        
        # Settings section
        st.header("⚙️ Settings")
        
        # Voice response toggle
        st.session_state.enable_voice_response = st.toggle(
            "🔊 Enable Voice Responses",
            value=st.session_state.enable_voice_response,
            help="When enabled, responses are also played as audio"
        )
        
        st.divider()
        
        # Knowledge Base section
        st.header("📚 Knowledge Base")
        
        # Show knowledge base status
        if st.session_state.retriever and st.session_state.retriever.is_ready():
            st.success("✓ Knowledge base loaded")
        else:
            st.warning("⚠️ Knowledge base empty - click 'Ingest Documents'")
        
        # Ingest documents button
        if st.button("📥 Ingest Documents", use_container_width=True):
            with st.spinner("Ingesting documents..."):
                try:
                    retriever = get_retriever()
                    count = retriever.ingest_documents()
                    if count > 0:
                        st.success(f"✓ Ingested {count} document chunks!")
                        st.balloons()
                    else:
                        st.warning("No documents found. Add files to data/support_docs/")
                except RetrieverError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")
        
        # Add custom document
        with st.expander("➕ Add Custom Document"):
            # Option 1: Upload file
            st.markdown("**📄 Upload Document:**")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["txt", "pdf", "docx", "md"],
                help="Upload TXT, PDF, DOCX, or Markdown files"
            )
            if uploaded_file is not None:
                if st.button("📤 Upload to Knowledge Base", use_container_width=True):
                    try:
                        # Read file content based on type
                        file_content = ""
                        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.md'):
                            file_content = uploaded_file.read().decode("utf-8")
                        elif uploaded_file.name.endswith('.pdf'):
                            try:
                                import PyPDF2
                                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                                for page in pdf_reader.pages:
                                    file_content += page.extract_text() + "\n"
                            except ImportError:
                                st.error("PDF support requires PyPDF2. Add 'PyPDF2' to requirements.txt")
                                file_content = ""
                        elif uploaded_file.name.endswith('.docx'):
                            try:
                                import docx
                                doc = docx.Document(uploaded_file)
                                file_content = "\n".join([para.text for para in doc.paragraphs])
                            except ImportError:
                                st.error("DOCX support requires python-docx. Add 'python-docx' to requirements.txt")
                                file_content = ""
                        
                        if file_content.strip():
                            retriever = get_retriever()
                            chunks = retriever.add_document(file_content)
                            if chunks > 0:
                                st.success(f"✓ Uploaded '{uploaded_file.name}' ({chunks} chunks)")
                            else:
                                st.warning("Document was empty")
                        else:
                            st.warning("Could not extract text from file.")
                    except Exception as e:
                        st.error(f"❌ Upload failed: {e}")
            
            st.divider()
            
            # Option 2: Paste content
            st.markdown("**📝 Or Paste Content:**")
            custom_doc = st.text_area(
                "Paste content:",
                height=100,
                placeholder="Paste FAQ, policy, or other support documentation..."
            )
            if st.button("Add to Knowledge Base", use_container_width=True):
                if custom_doc.strip():
                    retriever = get_retriever()
                    chunks = retriever.add_document(custom_doc)
                    if chunks > 0:
                        st.success(f"✓ Added document ({chunks} chunks)")
                    else:
                        st.warning("Document was empty")
                else:
                    st.warning("Please enter document content.")
        
        st.divider()
        
        # Conversation controls
        st.header("💬 Conversation")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.llm_handler:
                st.session_state.llm_handler.clear_history()
            st.rerun()
        
        st.divider()
        
        # System info
        st.header("ℹ️ System Info")
        config = get_config()
        st.caption(f"**STT:** Whisper ({config.WHISPER_MODEL})")
        st.caption("**LLM:** MockLLM (Zero-Cost)")
        st.caption("**TTS:** gTTS")
        st.caption("**Vector DB:** ChromaDB")
        st.caption(f"**Embeddings:** {config.EMBEDDING_MODEL}")
        
        st.divider()
        
        # Architecture note
        with st.expander("🏗️ Architecture"):
            st.markdown("""
            **Data Flow:**
            1. Voice → Whisper STT → Text
            2. Text → Embeddings → ChromaDB Search
            3. Context + Query → MockLLM → Response
            4. Response → gTTS → Audio
            
            **Why Mock LLM?**
            - Zero API costs
            - Works offline
            - Interview-safe demos
            - Easily swappable to real LLM
            """)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎙️ Voice RAG Customer Support</div>', unsafe_allow_html=True)
    st.caption("Ask questions via text or voice. Responses are grounded in customer support documentation.")
    
    # Sidebar
    sidebar_settings()
    
    # Load components
    if not load_components():
        st.error("Failed to initialize the application. Please refresh the page.")
        return
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("audio")
        )
    
    # Input tabs for clear separation of text and voice modes
    tab1, tab2 = st.tabs(["💬 Text Input", "🎤 Voice Input"])
    
    # ==========================================================================
    # TEXT INPUT TAB
    # ==========================================================================
    with tab1:
        # Text input field
        user_input = st.chat_input("Type your question here...")
        
        if user_input:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            display_chat_message("user", user_input)
            
            # Process query with status updates
            status_placeholder = st.empty()
            response = process_query(user_input, status_placeholder)
            
            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["text"],
                "audio": response.get("audio")
            })
            display_chat_message("assistant", response["text"], response.get("audio"))
            st.rerun()
    
    # ==========================================================================
    # VOICE INPUT TAB
    # ==========================================================================
    with tab2:
        st.subheader("🎤 Voice Input")
        
        # Check if STT is available
        if st.session_state.stt is None:
            st.warning(
                "⚠️ Voice input is not available.\n\n"
                "This could be because:\n"
                "- Whisper model failed to load\n"
                "- FFmpeg is not installed\n"
                "- Insufficient memory\n\n"
                "Please use text input instead."
            )
        else:
            st.caption("🎙️ Click the microphone to record your question, or upload an audio file.")
            
            # =================================================================
            # MICROPHONE RECORDING (Primary Option)
            # =================================================================
            st.markdown("##### 🎤 Record from Microphone")
            
            if AUDIO_RECORDER_AVAILABLE:
                audio_bytes = audio_recorder(
                    text="Click to record",
                    recording_color="#e74c3c",
                    neutral_color="#1E88E5",
                    icon_name="microphone",
                    icon_size="2x",
                    pause_threshold=2.0,
                    sample_rate=16000
                )
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    if st.button("🎯 Process Recording", use_container_width=True, key="process_mic"):
                        status_placeholder = st.empty()
                        
                        # Save audio bytes to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        try:
                            # Stage 1: Transcription
                            with status_placeholder.container():
                                st.info("🎤 Transcribing audio with Whisper...")
                            
                            with open(tmp_path, 'rb') as f:
                                transcribed_text, error = transcribe_audio(f)
                            
                            if error:
                                status_placeholder.empty()
                                st.error(f"❌ Transcription failed: {error}")
                            elif not transcribed_text or transcribed_text.startswith("[No"):
                                status_placeholder.empty()
                                st.warning(f"⚠️ {transcribed_text or 'No speech detected in the audio.'}")
                            else:
                                # Show transcription
                                st.success(f"📝 Transcribed: \"{transcribed_text}\"")
                                
                                # Add user message
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": f"🎤 {transcribed_text}"
                                })
                                
                                # Process the transcribed query
                                response = process_query(transcribed_text, status_placeholder)
                                
                                # Add assistant response
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response["text"],
                                    "audio": response.get("audio")
                                })
                                
                                st.rerun()
                        finally:
                            # Cleanup temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
            else:
                st.info("💡 Install `audio-recorder-streamlit` for microphone recording: `pip install audio-recorder-streamlit`")
            
            st.divider()
            
            # =================================================================
            # FILE UPLOAD (Fallback Option)
            # =================================================================
            st.markdown("##### 📁 Or Upload Audio File")
            
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
                key="audio_uploader"
            )
            
            if audio_file is not None:
                # Show audio preview
                st.audio(audio_file)
                
                if st.button("🎯 Transcribe & Process", use_container_width=True):
                    status_placeholder = st.empty()
                    
                    # Stage 1: Transcription
                    with status_placeholder.container():
                        st.info("🎤 Transcribing audio with Whisper...")
                    
                    transcribed_text, error = transcribe_audio(audio_file)
                    
                    if error:
                        status_placeholder.empty()
                        st.error(f"❌ Transcription failed: {error}")
                    elif not transcribed_text or transcribed_text.startswith("[No"):
                        status_placeholder.empty()
                        st.warning(f"⚠️ {transcribed_text or 'No speech detected in the audio.'}")
                    else:
                        # Show transcription
                        st.success(f"📝 Transcribed: \"{transcribed_text}\"")
                        
                        # Add user message
                        st.session_state.messages.append({
                            "role": "user",
                            "content": f"🎤 {transcribed_text}"
                        })
                        
                        # Process the transcribed query
                        response = process_query(transcribed_text, status_placeholder)
                        
                        # Add assistant response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["text"],
                            "audio": response.get("audio")
                        })
                        
                        st.rerun()
            
            st.divider()
            st.caption(
                "💡 **Tip:** Click the microphone button to record directly, "
                "or upload a pre-recorded audio file."
            )


if __name__ == "__main__":
    main()