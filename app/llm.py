"""
LLM (Large Language Model) module for response generation.

This module implements a model-agnostic LLM layer with:
- Abstract LLMInterface for swappable implementations
- Zero-cost MockLLM that generates responses from retrieved context
- Structured prompting for customer support use case

=============================================================================
MODULE RESPONSIBILITY (Single Responsibility Principle)
=============================================================================
This module ONLY handles:
✓ Defining the LLM interface contract
✓ Implementing response generation logic
✓ Managing conversation history

This module does NOT handle:
✗ Document retrieval (→ retriever.py)
✗ Audio processing (→ stt.py, tts.py)  
✗ User interface (→ streamlit_app.py)

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why abstract LLMInterface?"
   → Dependency Inversion Principle - high-level modules don't depend on
     low-level details. Makes swapping providers trivial.

2. "Why MockLLM instead of calling OpenAI?"
   → Zero-cost, interview-safe, demonstrates architecture without API keys

3. "How would you swap to GPT-4?"
   → Create OpenAILLM class implementing LLMInterface, update factory function

4. "What's the system prompt doing?"
   → Constrains LLM to use ONLY provided context, prevents hallucination

=============================================================================
DESIGN PATTERN: Strategy Pattern
=============================================================================
LLMInterface uses the Strategy pattern:
- Context: RAGLLMHandler (uses any LLM implementation)
- Strategy Interface: LLMInterface (defines generate_response contract)
- Concrete Strategies: MockLLM, (future) OpenAILLM, ClaudeLLM, etc.

=============================================================================
ZERO-COST ARCHITECTURE
=============================================================================
This implementation uses MockLLM by default to ensure:
1. ZERO external API costs (no OpenAI, Azure, Anthropic charges)
2. Fully offline operation (except initial Whisper model download)
3. Interview-safe demonstration without API key requirements
"""

import re
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Final
from dataclasses import dataclass

from app.config import get_config
from app.retriever import get_retriever


# =============================================================================
# CONSTANTS (Defensive Design)
# =============================================================================

# Maximum conversation history length to prevent memory bloat
MAX_HISTORY_LENGTH: Final[int] = 50

# Maximum query length to prevent abuse
MAX_QUERY_LENGTH: Final[int] = 2000

# Maximum context length for MockLLM processing
MAX_CONTEXT_LENGTH: Final[int] = 10000

# Minimum relevance score threshold for sentence inclusion  
MIN_RELEVANCE_SCORE: Final[int] = 3


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ChatMessage:
    """
    Represents a single chat message in the conversation.
    
    Immutable data class ensuring message integrity.
    
    Attributes:
        role: Who sent the message ("user", "assistant", or "system")
        content: The actual message text (never empty)
    
    INTERVIEW NOTE:
    Using @dataclass provides:
    - Automatic __init__, __repr__, __eq__
    - Type hints for IDE support
    - Immutability when frozen=True (not used here for simplicity)
    """
    role: str
    content: str
    
    def __post_init__(self):
        """Validate message after creation."""
        valid_roles = {"user", "assistant", "system"}
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of: {valid_roles}")
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")


# =============================================================================
# SYSTEM PROMPT - Used even in Mock Mode for structured responses
# =============================================================================

CUSTOMER_SUPPORT_SYSTEM_PROMPT = """You are a helpful and friendly customer support assistant.

YOUR ROLE:
1. Answer customer questions accurately based ONLY on the provided context documents
2. Be polite, professional, and empathetic in your responses
3. Keep responses concise but complete
4. Never make up information that is not present in the context

CRITICAL RULES:
- ONLY use information from the provided context documents
- If the context doesn't contain relevant information, say: "I don't have specific information about that in our documentation. Please contact our support team for further assistance."
- If you're unsure about something, acknowledge it rather than guessing
- For complex issues not covered in documentation, suggest contacting human support

RESPONSE FORMAT:
- Start with a direct answer when possible
- Use bullet points for multiple items
- End with an offer to help further if appropriate"""


# =============================================================================
# ABSTRACT LLM INTERFACE
# =============================================================================

class LLMInterface(ABC):
    """
    Abstract base class defining the LLM interface.
    
    This abstraction allows easy swapping between:
    - MockLLM (zero-cost, offline)
    - OpenAI GPT models
    - Local models (Llama, Mistral, etc.)
    - Any other LLM provider
    
    All implementations must provide:
    - generate_response(): Main method for generating responses
    - clear_history(): Reset conversation state
    - get_history(): Retrieve conversation history
    """
    
    @abstractmethod
    def generate_response(
        self, 
        query: str,
        context: str = "",
        include_history: bool = True
    ) -> str:
        """
        Generate a response to the user query.
        
        Args:
            query: User's question or message
            context: Retrieved context from RAG (may be empty)
            include_history: Whether to consider conversation history
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def clear_history(self) -> None:
        """Clear the conversation history."""
        pass
    
    @abstractmethod
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history as list of dicts."""
        pass


# =============================================================================
# MOCK LLM IMPLEMENTATION (Zero-Cost)
# =============================================================================

class MockLLM(LLMInterface):
    """
    Zero-cost LLM implementation that generates responses from retrieved context.
    
    ALGORITHM OVERVIEW:
    1. Receives the user query and retrieved context
    2. Splits context into sentences
    3. Scores each sentence by relevance to query keywords
    4. Selects top-scoring sentences for response
    5. Formats into customer support tone
    
    WHY MOCK LLM (Interview Talking Point):
    - Zero API costs - completely free to run
    - Works offline - no internet required after setup
    - Interview-safe - demonstrates architecture without API keys
    - Predictable - responses are based directly on documents
    - Testable - deterministic output for unit testing
    
    LIMITATIONS (Be Honest in Interviews):
    - Cannot paraphrase or synthesize information creatively
    - Responses may feel repetitive for similar queries
    - No true \"understanding\" - keyword matching only
    - Would swap to real LLM for production
    
    COMPLEXITY: O(n * m) where n = sentences, m = query words
    """
    
    def __init__(self):
        """
        Initialize the MockLLM with empty conversation history.
        
        DESIGN: Lazy initialization - no heavy operations in constructor.
        """
        # Store conversation history for context continuity
        self.conversation_history: List[ChatMessage] = []
        
        # Topic keywords for relevance scoring
        # DESIGN: Easily extensible - add new topics as needed
        self.topic_keywords: Dict[str, List[str]] = {
            'return': ['return', 'refund', 'exchange', 'money back'],
            'shipping': ['shipping', 'delivery', 'ship', 'arrive', 'track'],
            'password': ['password', 'reset', 'login', 'sign in', 'forgot'],
            'payment': ['payment', 'pay', 'billing', 'charge', 'card', 'credit'],
            'cancel': ['cancel', 'subscription', 'unsubscribe'],
            'contact': ['contact', 'support', 'help', 'phone', 'email'],
        }
    
    def generate_response(
        self, 
        query: str,
        context: str = "",
        include_history: bool = True
    ) -> str:
        """
        Generate a response based on retrieved context.
        
        ALGORITHM:
        1. Validate and sanitize inputs
        2. Check if context is available
        3. Extract relevant sentences using keyword scoring
        4. Format into customer-friendly response
        
        DEFENSIVE DESIGN:
        - Handles empty/None inputs gracefully
        - Truncates excessively long inputs
        - Maintains bounded conversation history
        
        Args:
            query: User's question (will be sanitized)
            context: Retrieved document context (from RAG)
            include_history: Whether to use conversation history
            
        Returns:
            A response string grounded in the context
        """
        # DEFENSIVE: Sanitize query input
        query = (query or "").strip()
        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]
        
        if not query:
            return "I didn't receive a question. How can I help you today?"
        
        # DEFENSIVE: Truncate excessively long context
        context = (context or "").strip()
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH]
        
        # Store the user's query in history
        self.conversation_history.append(ChatMessage(role="user", content=query))
        self._trim_history()  # DEFENSIVE: Prevent memory bloat
        
        # Case 1: No context available - knowledge base is empty or retrieval failed
        if not context:
            response = self._generate_no_context_response()
            self.conversation_history.append(ChatMessage(role="assistant", content=response))
            return response
        
        # Case 2: Context indicates no relevant information found
        if "No relevant information" in context:
            response = self._generate_insufficient_info_response(query)
            self.conversation_history.append(ChatMessage(role="assistant", content=response))
            return response
        
        # Case 3: We have context - extract relevant information
        response = self._generate_grounded_response(query, context)
        self.conversation_history.append(ChatMessage(role="assistant", content=response))
        return response
    
    def _generate_no_context_response(self) -> str:
        """Generate response when no knowledge base is available."""
        return (
            "I apologize, but I don't have access to any documentation at the moment. "
            "Please ensure the knowledge base has been loaded by clicking 'Ingest Documents' "
            "in the sidebar. If you need immediate assistance, please contact our support team."
        )
    
    def _generate_insufficient_info_response(self, query: str) -> str:
        """Generate response when context doesn't contain relevant info."""
        return (
            f"I understand you're asking about: \"{query}\"\n\n"
            "Unfortunately, I don't have specific information about that in our documentation. "
            "For questions not covered in our FAQ or policies, I recommend:\n"
            "• Contacting our support team directly\n"
            "• Calling 1-800-TECH-PRO (8am-8pm EST, Monday-Friday)\n"
            "• Using live chat on our website\n\n"
            "Is there anything else I can help you with?"
        )
    
    def _generate_grounded_response(self, query: str, context: str) -> str:
        """
        Generate a response grounded in the retrieved context.
        
        This method:
        1. Splits context into sentences
        2. Finds sentences relevant to the query
        3. Combines them into a coherent response
        4. Adds appropriate formatting
        """
        # Normalize the query for matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Split context into sentences for analysis
        sentences = self._split_into_sentences(context)
        
        # Score each sentence by relevance to the query
        scored_sentences = []
        for sentence in sentences:
            score = self._calculate_relevance_score(sentence, query_lower, query_words)
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by relevance score (highest first)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Take top relevant sentences (up to 5)
        top_sentences = [s[0] for s in scored_sentences[:5]]
        
        # If no relevant sentences found, provide a general response
        if not top_sentences:
            return self._generate_general_context_response(context)
        
        # Build the response
        response_parts = ["Based on our documentation:\n"]
        
        # Add relevant information
        for sentence in top_sentences:
            # Clean up the sentence
            clean_sentence = sentence.strip()
            if clean_sentence and not clean_sentence.startswith('['):
                response_parts.append(f"• {clean_sentence}")
        
        # Add helpful closing
        response_parts.append("\nIs there anything else you'd like to know?")
        
        return "\n".join(response_parts)
    
    def _generate_general_context_response(self, context: str) -> str:
        """Generate response when we have context but couldn't find specific matches."""
        # Extract first meaningful paragraph from context
        paragraphs = context.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50 and not para.startswith('['):
                preview = para[:300].strip()
                if not preview.endswith('.'):
                    # Find last complete sentence
                    last_period = preview.rfind('.')
                    if last_period > 0:
                        preview = preview[:last_period + 1]
                
                return (
                    f"Here's what I found in our documentation:\n\n{preview}\n\n"
                    "Would you like more specific information about any topic?"
                )
        
        return (
            "I found some information in our documentation, but I'm not sure it directly "
            "answers your question. Could you please rephrase or be more specific about "
            "what you'd like to know?"
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for analysis."""
        # Remove document markers
        text = re.sub(r'\[Document \d+\][^\n]*', '', text)
        text = re.sub(r'---+', '', text)
        text = re.sub(r'\(Source:[^\)]*\)', '', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences and clean up
        return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def _calculate_relevance_score(
        self, 
        sentence: str, 
        query_lower: str, 
        query_words: set
    ) -> int:
        """
        Calculate how relevant a sentence is to the query.
        
        Scoring:
        - +3 points for each query word found in the sentence
        - +5 points for topic keyword matches
        - +2 points for question word matches (how, what, when, etc.)
        
        COMPLEXITY: O(w + t*k) where w = query words, t = topics, k = keywords per topic
        """
        sentence_lower = sentence.lower()
        score = 0
        
        # Score based on query word matches
        for word in query_words:
            if len(word) > 2 and word in sentence_lower:
                score += 3
        
        # Bonus for topic keyword matches
        for topic, keywords in self.topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if any(kw in sentence_lower for kw in keywords):
                    score += 5
        
        return score
    
    def _trim_history(self) -> None:
        """
        Trim conversation history to prevent memory bloat.
        
        DEFENSIVE DESIGN: Unbounded history could cause memory issues
        in long-running sessions. We keep only the last MAX_HISTORY_LENGTH messages.
        """
        if len(self.conversation_history) > MAX_HISTORY_LENGTH:
            # Keep most recent messages, drop oldest
            self.conversation_history = self.conversation_history[-MAX_HISTORY_LENGTH:]
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as list of dictionaries.
        
        Returns a COPY to prevent external mutation of internal state.
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history
        ]


# =============================================================================
# RAG-ENABLED LLM HANDLER
# =============================================================================

class RAGLLMHandler:
    """
    Wrapper that combines RAG retrieval with LLM response generation.
    
    This class orchestrates:
    1. Query processing
    2. Context retrieval from ChromaDB
    3. Response generation via the LLM interface
    4. Conversation history management
    
    ARCHITECTURE:
    User Query → Retriever (ChromaDB) → Context → LLM → Response
    """
    
    def __init__(self, llm: LLMInterface):
        """
        Initialize the RAG-enabled handler.
        
        Args:
            llm: An implementation of LLMInterface (e.g., MockLLM)
        """
        self.llm = llm
        self.retriever = get_retriever()
    
    def generate_response(
        self, 
        query: str,
        use_rag: bool = True,
        include_history: bool = True
    ) -> str:
        """
        Generate a response using RAG pipeline.
        
        Steps:
        1. If RAG enabled, retrieve relevant context from knowledge base
        2. Pass query + context to the LLM
        3. Return the generated response
        
        Args:
            query: User's question
            use_rag: Whether to retrieve context (default: True)
            include_history: Whether to use conversation history
            
        Returns:
            Generated response string
        """
        # Step 1: Retrieve context if RAG is enabled
        context = ""
        if use_rag:
            try:
                context = self.retriever.get_context(query)
            except Exception as e:
                # Log error but continue without context
                print(f"Warning: Context retrieval failed: {e}")
                context = ""
        
        # Step 2: Generate response using the LLM
        response = self.llm.generate_response(
            query=query,
            context=context,
            include_history=include_history
        )
        
        return response
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.llm.clear_history()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.llm.get_history()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_llm_handler(use_mock: bool = True) -> RAGLLMHandler:
    """
    Factory function to create an LLM handler.
    
    CURRENT IMPLEMENTATION:
    Always returns MockLLM for zero-cost operation.
    
    TO USE A REAL LLM:
    1. Create a class implementing LLMInterface
    2. Modify this function to instantiate your implementation
    3. Example:
        if not use_mock and api_key:
            llm = OpenAILLM(api_key=api_key)
        else:
            llm = MockLLM()
    
    Args:
        use_mock: Ignored in zero-cost mode (always True)
        
    Returns:
        RAGLLMHandler configured with MockLLM
    """
    # Zero-cost design: Always use MockLLM
    # This ensures no API charges regardless of the use_mock flag
    llm = MockLLM()
    
    return RAGLLMHandler(llm=llm)


# =============================================================================
# EXAMPLE: How to implement a real LLM (commented out)
# =============================================================================

# class OpenAILLM(LLMInterface):
#     """Example implementation using OpenAI API."""
#     
#     def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
#         from openai import OpenAI
#         self.client = OpenAI(api_key=api_key)
#         self.model = model
#         self.conversation_history: List[ChatMessage] = []
#     
#     def generate_response(self, query: str, context: str = "", include_history: bool = True) -> str:
#         messages = [{"role": "system", "content": CUSTOMER_SUPPORT_SYSTEM_PROMPT}]
#         
#         # Add context to the user message
#         user_content = f"Context:\n{context}\n\nQuestion: {query}" if context else query
#         messages.append({"role": "user", "content": user_content})
#         
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             temperature=0.3,
#             max_tokens=512
#         )
#         
#         return response.choices[0].message.content
#     
#     def clear_history(self) -> None:
#         self.conversation_history = []
#     
#     def get_history(self) -> List[Dict[str, str]]:
#         return [{"role": m.role, "content": m.content} for m in self.conversation_history]
