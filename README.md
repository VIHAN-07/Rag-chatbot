# 🎙️ Voice-Enabled RAG Chatbot for Customer Support

A production-style, interview-ready conversational AI system that supports both text and voice input/output, grounded in domain-specific customer support documents using Retrieval-Augmented Generation (RAG).

## � ZERO-COST DESIGN

**This application runs with ZERO external API costs.** No OpenAI, Azure, Anthropic, or any paid API is required.

| Component | Cost | How |
|-----------|------|-----|
| LLM | **FREE** | MockLLM generates responses from retrieved context |
| Embeddings | **FREE** | HuggingFace model runs locally |
| STT | **FREE** | Whisper runs locally |
| TTS | **FREE** | gTTS uses free Google service |
| Vector DB | **FREE** | ChromaDB runs embedded |

## 📋 Table of Contents

- [Overview](#overview)
- [Design Decisions](#design-decisions)
- [Why Mock LLM / Zero-Cost Design](#why-mock-llm--zero-cost-design)
- [How to Swap in a Real LLM](#how-to-swap-in-a-real-llm)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Resume Alignment](#resume-alignment)
- [Limitations](#limitations)

## 🎯 Overview

This project implements a modular, voice-enabled customer support chatbot that:

- **Accepts text and voice input** from users
- **Converts speech to text** using OpenAI Whisper (runs locally)
- **Retrieves relevant context** from customer support documents using ChromaDB
- **Generates grounded responses** using MockLLM with RAG
- **Converts responses to speech** using Text-to-Speech
- **Provides a clean Streamlit UI** for interaction

### Key Technologies

| Component | Technology | Cost |
|-----------|------------|------|
| Speech-to-Text (STT) | OpenAI Whisper (local) | Free |
| Vector Database | ChromaDB (embedded) | Free |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Free |
| RAG Framework | LangChain | Free |
| LLM | **MockLLM** (swappable) | Free |
| Text-to-Speech (TTS) | gTTS | Free |
| UI Framework | Streamlit | Free |

## 🎨 Design Decisions

### 1. Model-Agnostic LLM Layer

The LLM module uses an **abstract interface pattern** (`LLMInterface`) that allows easy swapping between implementations:

```python
class LLMInterface(ABC):
    @abstractmethod
    def generate_response(self, query: str, context: str = "") -> str:
        pass
```

**Why this matters:**
- Demonstrates understanding of clean architecture principles
- Interview-safe: shows abstraction and dependency injection
- Production-ready: swap MockLLM for OpenAI/Anthropic without code changes

### 2. RAG for Grounding (Not Hallucination)

**Why RAG instead of fine-tuning?**
- No training data or GPU compute required
- Easily updated by adding new documents
- Responses are grounded in actual documentation
- More explainable and auditable for customer support

**Why ChromaDB?**
- Lightweight, embedded vector database
- No external server needed
- Perfect for prototyping and small-to-medium datasets

### 3. Document Chunking Strategy

Documents are split into chunks because:
- Embedding models have token limits (~512 tokens)
- Smaller chunks enable more precise retrieval
- Overlap (50 chars) prevents information loss at boundaries

### 4. Non-Streaming Pipeline

**Why batch processing instead of streaming?**
- Simpler architecture, easier to debug
- Each component can be tested independently
- Meets accuracy requirements for customer support
- Designed for future streaming extensibility

## 💡 Why Mock LLM / Zero-Cost Design

### The Problem with API-Based LLMs

| Issue | Impact |
|-------|--------|
| **Cost** | OpenAI charges ~$0.002/1K tokens. Heavy testing gets expensive. |
| **API Keys** | Requires exposing API keys in demos/interviews |
| **Internet** | Doesn't work offline |
| **Rate Limits** | Can fail during live demos |

### How MockLLM Works

The MockLLM generates responses by:

1. **Receiving the user query and retrieved context** from ChromaDB
2. **Analyzing context for relevant sentences** using keyword matching
3. **Scoring sentences by relevance** to the query
4. **Constructing a response** using top-scoring sentences
5. **Following customer support tone** (polite, professional)
6. **Clearly indicating** when information is not available

```python
# Example flow in MockLLM:
query = "How do I reset my password?"
context = "[Retrieved from ChromaDB: password reset instructions...]"

# MockLLM extracts relevant sentences and formats response:
response = """Based on our documentation:
• To reset your password, click the "Forgot Password" link on the login page
• Enter your registered email address
• You'll receive a password reset link within 5 minutes

Is there anything else you'd like to know?"""
```

### Interview Benefits

| Benefit | Explanation |
|---------|-------------|
| **No API key needed** | Demo works anywhere, anytime |
| **Predictable responses** | Based directly on documents |
| **Fully offline** | Works without internet (after initial setup) |
| **Shows RAG understanding** | Responses are clearly grounded in context |

## 🔄 How to Swap in a Real LLM

The architecture is designed for easy LLM replacement. Here's how:

### Step 1: Implement LLMInterface

```python
# In app/llm.py

from openai import OpenAI

class OpenAILLM(LLMInterface):
    """Real LLM implementation using OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
    
    def generate_response(
        self, 
        query: str, 
        context: str = "", 
        include_history: bool = True
    ) -> str:
        # Build messages with system prompt
        messages = [{
            "role": "system", 
            "content": CUSTOMER_SUPPORT_SYSTEM_PROMPT
        }]
        
        # Add context to user message
        user_content = f"Context:\n{context}\n\nQuestion: {query}"
        messages.append({"role": "user", "content": user_content})
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=512
        )
        
        return response.choices[0].message.content
    
    def clear_history(self):
        self.conversation_history = []
    
    def get_history(self):
        return [{"role": m.role, "content": m.content} 
                for m in self.conversation_history]
```

### Step 2: Update Factory Function

```python
# In app/llm.py

def get_llm_handler(use_mock: bool = True) -> RAGLLMHandler:
    if use_mock:
        llm = MockLLM()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for real LLM")
        llm = OpenAILLM(api_key=api_key)
    
    return RAGLLMHandler(llm=llm)
```

### Step 3: Add Environment Variable

```bash
export OPENAI_API_KEY=your-api-key-here
```

### Other LLM Options

| Provider | Implementation Notes |
|----------|---------------------|
| **Anthropic Claude** | Similar pattern, use `anthropic` package |
| **Local Llama** | Use `llama-cpp-python` or `ollama` |
| **Azure OpenAI** | Use `AzureOpenAI` client |
| **Hugging Face** | Use `transformers` or `text-generation-inference` |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                                 │
│  ┌──────────────────┐          ┌──────────────────────────────────┐ │
│  │   Text Input     │          │         Voice Input               │ │
│  │   (Chat Box)     │          │    (Audio File Upload)            │ │
│  └────────┬─────────┘          └────────────────┬─────────────────┘ │
└───────────┼─────────────────────────────────────┼───────────────────┘
            │                                     │
            │                                     ▼
            │                          ┌──────────────────┐
            │                          │   STT Module     │
            │                          │   (Whisper)      │
            │                          └────────┬─────────┘
            │                                   │
            ▼                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TEXT QUERY                                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RETRIEVER MODULE                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Query         │───▶│   ChromaDB      │───▶│   Relevant      │ │
│  │   Embedding     │    │   Search        │    │   Context       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LLM MODULE (MockLLM - Zero Cost)              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Context +     │───▶│   MockLLM       │───▶│   Grounded      │ │
│  │   Query         │    │   (Local)       │    │   Response      │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TTS MODULE                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   Text          │───▶│   gTTS          │───▶│   Audio         │ │
│  │   Response      │    │   Synthesis     │    │   Output        │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                                 │
│  ┌──────────────────────────┐  ┌──────────────────────────────────┐ │
│  │   Text Response Display  │  │   Audio Playback Component       │ │
│  └──────────────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| **Config** | `app/config.py` | Centralized configuration management |
| **STT** | `app/stt.py` | Speech-to-Text using Whisper |
| **Retriever** | `app/retriever.py` | Document ingestion, embedding, and retrieval |
| **LLM** | `app/llm.py` | Response generation with RAG grounding |
| **TTS** | `app/tts.py` | Text-to-Speech audio generation |
| **UI** | `streamlit_app.py` | User interface and interaction flow |

## 🔄 Data Flow

### Text Input Path

```
User Types Question
        │
        ▼
┌───────────────────┐
│  Query Received   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     ┌───────────────────┐
│  Generate Query   │────▶│  Search ChromaDB  │
│  Embedding        │     │  (Top-K Results)  │
└───────────────────┘     └─────────┬─────────┘
                                    │
          ┌─────────────────────────┘
          │
          ▼
┌───────────────────┐     ┌───────────────────┐
│  Build Prompt     │────▶│  MockLLM          │
│  (Query+Context)  │     │  (Zero-Cost)      │
└───────────────────┘     └─────────┬─────────┘
                                    │
          ┌─────────────────────────┘
          │
          ▼
┌───────────────────┐     ┌───────────────────┐
│  Text Response    │────▶│  TTS Synthesis    │
│  Generated        │     │  (Audio Output)   │
└───────────────────┘     └─────────┬─────────┘
                                    │
                                    ▼
                          Display Text + Play Audio
```

### Voice Input Path

```
User Uploads Audio File
        │
        ▼
┌───────────────────┐
│  Whisper STT      │
│  Transcription    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Transcribed Text │──────▶ [Same as Text Path above]
└───────────────────┘
```

### RAG Pipeline Details

1. **Document Ingestion**
   - Load documents from `data/support_docs/`
   - Split into chunks (500 chars, 50 overlap)
   - Generate embeddings using SentenceTransformers
   - Store in ChromaDB vector database

2. **Query Processing**
   - Convert user query to embedding
   - Perform similarity search in ChromaDB
   - Retrieve top-K most relevant chunks

3. **Response Generation**
   - Combine query with retrieved context
   - Apply system prompt for customer support persona
   - Generate response using LLM (inference only, no fine-tuning)

## 📁 Project Structure

```
voice_rag_chatbot/
├── app/
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration settings
│   ├── stt.py            # Speech-to-Text (Whisper)
│   ├── retriever.py      # RAG retrieval (ChromaDB + LangChain)
│   ├── llm.py            # LLM response generation
│   └── tts.py            # Text-to-Speech
├── data/
│   └── support_docs/     # Customer support documents
│       ├── sample_faq.txt
│       ├── shipping_policy.txt
│       └── return_policy.txt
├── streamlit_app.py      # Streamlit UI
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg (required for Whisper audio processing)
- **No API keys required** (zero-cost design using MockLLM)

### Step 1: Clone and Navigate

```bash
cd voice_rag_chatbot
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install FFmpeg (Required for Whisper)

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**Windows (using Winget):**
```bash
winget install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

### Step 5: Run the Application

```bash
streamlit run streamlit_app.py
```

The application will open at `http://localhost:8501`. No API keys required!

## 💻 Usage

### Start the Application

```bash
streamlit run streamlit_app.py
```

The application will open at `http://localhost:8501`

### Using the Chatbot

1. **Ingest Documents**: Click "Ingest Documents" in the sidebar to load the sample customer support documents into ChromaDB.

2. **Text Chat**: Type your question in the chat input box and press Enter.

3. **Voice Input**: 
   - Switch to the "Voice Input" tab
   - Upload an audio file (.wav, .mp3, .m4a, .flac)
   - Click "Transcribe & Process"

4. **Voice Responses**: Audio playback is automatically generated for each response (can be toggled off in settings).

> **Note**: This implementation uses MockLLM by default (zero external API cost). See "How to Swap in a Real LLM" section above to use OpenAI, Claude, or other providers.

## ⚙️ Configuration

Edit `app/config.py` to customize:

```python
@dataclass
class Config:
    # Model settings
    WHISPER_MODEL: str = "base"      # tiny, base, small, medium, large
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local embedding model
    
    # RAG settings
    CHUNK_SIZE: int = 500            # Document chunk size
    CHUNK_OVERLAP: int = 50          # Overlap between chunks
    TOP_K_RESULTS: int = 3           # Number of docs to retrieve
    
    # LLM settings (used when swapping to real LLM)
    LLM_TEMPERATURE: float = 0.3     # Response creativity (0-1)
    LLM_MAX_TOKENS: int = 512        # Max response length
```

> **Note**: By default, MockLLM is used (zero-cost). Temperature and max_tokens settings apply when you swap in a real LLM provider.

## 📝 Resume Alignment

This implementation directly supports the following resume claims:

| Resume Claim | Implementation |
|--------------|----------------|
| **Conversational AI with text + voice I/O** | Streamlit UI with text chat and voice upload/playback |
| **Whisper-based STT integration** | `app/stt.py` - OpenAI Whisper (runs locally, zero-cost) |
| **TTS-based audio response generation** | `app/tts.py` - gTTS for audio synthesis |
| **RAG using ChromaDB** | `app/retriever.py` - ChromaDB + LangChain retrieval |
| **Modular, low-latency pipeline** | Clear separation: STT → RAG → LLM → TTS modules |
| **Abstract LLM interface** | `app/llm.py` - LLMInterface ABC for swappable providers |
| **Zero-cost design** | MockLLM + local Whisper + local embeddings |
| **Extensible to real-time voice agents** | Architecture designed for future streaming integration |

### Interview Discussion Points

1. **Why Zero-Cost Design?** 
   - Demonstrates architecture without API dependencies
   - Interview-safe: runs anywhere without keys
   - Shows understanding of abstraction patterns

2. **Why ChromaDB?** 
   - Lightweight, embedded vector database
   - Easy setup, no external server required
   - Good for prototyping and small-to-medium datasets

3. **Why MockLLM with Abstract Interface?**
   - Demonstrates interface-based design patterns
   - Easy to swap providers (OpenAI, Anthropic, local models)
   - Testable without external dependencies

4. **Why RAG over Fine-tuning?**
   - No training data or compute required
   - Easily updated by adding new documents
   - Responses grounded in actual documentation
   - More explainable and auditable

## ⚠️ Limitations

### Current Limitations

| Limitation | Reason | Mitigation |
|------------|--------|------------|
| **MockLLM responses** | Zero-cost design, no API | Swappable to real LLM |
| **Non-streaming** | Batch processing pipeline | Designed for extensibility to streaming |
| **No real-time voice** | Requires WebRTC/telephony | Focus on accuracy over latency |
| **Single language** | English only | Easily extensible to other languages |
| **Local Whisper** | Requires FFmpeg, compute | Can swap to Whisper API |
| **No multi-turn context retrieval** | RAG per query only | Could add conversation-aware retrieval |

### Not Implemented (By Design)

- Paid API integration (by design - zero-cost)
- Real-time streaming (WebRTC, WebSocket)
- Telephony integration (Twilio, VoIP)
- Model fine-tuning (inference + prompt engineering only)
- Multi-user authentication
- Production deployment configurations (Docker, K8s)

## 🔮 Future Extensibility

This architecture is designed to be extended for real-time contact-center voice agents:

### Streaming Architecture (Not Implemented)

```
┌─────────────────────────────────────────────────────────────────┐
│                   FUTURE: REAL-TIME PIPELINE                     │
│                                                                  │
│  WebRTC Audio ──▶ Streaming STT ──▶ RAG ──▶ Streaming LLM       │
│       │                                          │               │
│       │                                          ▼               │
│       ◀────────────────── Streaming TTS ◀────────┘               │
│                                                                  │
│  Latency Target: <500ms end-to-end                              │
└─────────────────────────────────────────────────────────────────┘
```

### Extension Points

1. **Replace gTTS with streaming TTS** (e.g., ElevenLabs, Play.ht)
2. **Add WebSocket for real-time communication**
3. **Implement conversation-aware retrieval** (multi-turn context)
4. **Add telephony adapters** (Twilio, Vonage)
5. **Deploy as containerized microservices**

## 📄 License

MIT License - feel free to use for learning and portfolio purposes.

## 🤝 Contributing

This is a portfolio/interview project. Feel free to fork and adapt for your own use.

---

Built with ❤️ for demonstrating production-style GenAI development.
