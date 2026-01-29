"""
Document Retriever module using ChromaDB and LangChain.

This module implements Retrieval-Augmented Generation (RAG) for grounding 
chatbot responses in actual customer support documentation.

=============================================================================
MODULE RESPONSIBILITY (Single Responsibility Principle)
=============================================================================
This module ONLY handles:
✓ Document ingestion and chunking
✓ Embedding generation and storage  
✓ Similarity search and context retrieval

This module does NOT handle:
✗ Response generation (→ llm.py)
✗ Audio processing (→ stt.py, tts.py)
✗ User interface (→ streamlit_app.py)

=============================================================================
INTERVIEW TALKING POINTS
=============================================================================
1. "Why RAG instead of fine-tuning?"
   → No training data needed, easily updated, responses grounded in docs

2. "Why ChromaDB over Pinecone/Weaviate?"
   → Embedded (no server), zero-cost, good for prototyping

3. "Why HuggingFace embeddings over OpenAI?"
   → Free, offline, avoids API costs

4. "What's the chunk size tradeoff?"
   → Smaller = precise but may lose context; Larger = context-rich but noisy

=============================================================================
COMPLEXITY ANALYSIS (Interview-Ready)
=============================================================================
| Operation          | Time Complexity | Space Complexity |
|--------------------|-----------------|------------------|
| ingest_documents() | O(n * d)        | O(n)             |
| retrieve()         | O(log n) approx | O(k)             |
| get_context()      | O(log n + k)    | O(k)             |

Where: n = number of chunks, d = embedding dimension (384), k = top_k results

=============================================================================
RAG PIPELINE OVERVIEW
=============================================================================
1. INGESTION: Documents → Chunks → Embeddings → ChromaDB
2. RETRIEVAL: Query → Embedding → Similarity Search → Top-K Chunks  
3. GROUNDING: Chunks + Query → LLM → Grounded Response
"""

import os
from typing import List, Optional, Tuple, Final
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config import get_config


# =============================================================================
# CONSTANTS (Defensive Design - Avoid Magic Values)
# =============================================================================

# Supported document file extensions
SUPPORTED_DOC_FORMATS: Final[set] = {".txt", ".md"}

# Minimum document length to avoid indexing empty files
MIN_DOCUMENT_LENGTH: Final[int] = 10

# Maximum query length to prevent abuse
MAX_QUERY_LENGTH: Final[int] = 2000

# Default context message when retrieval returns empty
NO_CONTEXT_MESSAGE: Final[str] = "No relevant information found in the knowledge base."


# =============================================================================
# CUSTOM EXCEPTIONS FOR CLEAR ERROR HANDLING
# =============================================================================

class RetrieverError(Exception):
    """Base exception for retriever errors."""
    pass


class DocumentIngestionError(RetrieverError):
    """Raised when document ingestion fails."""
    pass


class DocumentNotFoundError(RetrieverError):
    """Raised when no documents are found to ingest."""
    pass


class EmptyKnowledgeBaseError(RetrieverError):
    """Raised when querying an empty knowledge base."""
    pass


# =============================================================================
# DOCUMENT RETRIEVER CLASS
# =============================================================================

class DocumentRetriever:
    """
    Document retriever using ChromaDB for vector storage.
    
    This class handles the complete RAG pipeline:
    
    1. DOCUMENT INGESTION (ingest_documents)
       - Load documents from the support_docs folder
       - Split into chunks for better retrieval granularity
       - Generate embeddings using HuggingFace model
       - Store in ChromaDB vector database
    
    2. QUERY-TIME RETRIEVAL (retrieve, get_context)
       - Convert user query to embedding
       - Find similar document chunks in ChromaDB
       - Return relevant context for LLM grounding
    
    DESIGN DECISIONS:
    - HuggingFace embeddings: Free, works offline, good quality
    - ChromaDB: Simple, embedded, no external server needed
    - Recursive chunking: Respects paragraph boundaries
    """
    
    def __init__(self):
        """
        Initialize the document retriever with ChromaDB.
        
        Sets up:
        - Embedding model (HuggingFace all-MiniLM-L6-v2)
        - Text splitter with overlap for better chunking
        - Connection to ChromaDB (creates or loads existing)
        """
        self.config = get_config()
        
        # Initialize the embedding model
        # This converts text to vectors for similarity search
        self.embeddings = self._init_embeddings()
        
        # Vector store reference (None until documents are ingested)
        self.vector_store: Optional[Chroma] = None
        
        # Text splitter configuration
        # Chunks are 500 chars with 50 char overlap to avoid cutting sentences
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,       # Max size of each chunk
            chunk_overlap=self.config.CHUNK_OVERLAP, # Overlap prevents info loss
            length_function=len,
            # Priority order for splitting: paragraphs > lines > sentences > words
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Try to load existing vector store from disk
        self._load_or_create_vector_store()
    
    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initialize the embedding model.
        
        Uses HuggingFace's all-MiniLM-L6-v2 model because:
        - Free to use (no API costs)
        - Works offline after initial download
        - Good balance of quality and speed
        - Produces 384-dimensional vectors
        
        Returns:
            Configured HuggingFaceEmbeddings instance
        """
        print("Initializing embedding model...")
        return HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Better for cosine similarity
        )
    
    def _load_or_create_vector_store(self) -> None:
        """
        Load existing vector store or prepare for new one.
        
        ChromaDB persists data to disk, so we check if a previous
        database exists and load it. This avoids re-ingesting documents
        on every application restart.
        """
        persist_dir = self.config.CHROMA_PERSIST_DIR
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print("Loading existing ChromaDB vector store...")
            try:
                self.vector_store = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.config.COLLECTION_NAME
                )
                print("Vector store loaded successfully.")
            except Exception as e:
                print(f"Warning: Could not load existing vector store: {e}")
                print("Will create new vector store on first ingestion.")
                self.vector_store = None
        else:
            print("No existing vector store found. Click 'Ingest Documents' to create one.")
    
    # =========================================================================
    # DOCUMENT INGESTION PHASE
    # =========================================================================
    
    def ingest_documents(self, docs_path: Optional[str] = None) -> int:
        """
        Ingest documents from the specified directory into ChromaDB.
        
        This is the INGESTION phase of RAG:
        1. Load all .txt and .md files from the documents folder
        2. Split each document into smaller chunks
        3. Generate embedding vectors for each chunk
        4. Store chunks + embeddings in ChromaDB
        
        Args:
            docs_path: Path to documents directory. Uses config default if None.
            
        Returns:
            Number of document chunks successfully ingested
            
        Raises:
            DocumentNotFoundError: If no documents found in the directory
            DocumentIngestionError: If ingestion fails
        """
        docs_path = docs_path or self.config.SUPPORT_DOCS_DIR
        docs_path = Path(docs_path)
        
        # Validate directory exists
        if not docs_path.exists():
            raise DocumentNotFoundError(
                f"Documents directory not found: {docs_path}\n"
                "Please create the folder and add support documents."
            )
        
        documents = []
        loaded_files = []
        errors = []
        
        # Load text files (.txt)
        for txt_file in docs_path.glob("*.txt"):
            try:
                loader = TextLoader(str(txt_file), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                loaded_files.append(txt_file.name)
                print(f"✓ Loaded: {txt_file.name}")
            except Exception as e:
                errors.append(f"{txt_file.name}: {e}")
                print(f"✗ Error loading {txt_file.name}: {e}")
        
        # Load markdown files (.md)
        for md_file in docs_path.glob("*.md"):
            try:
                loader = TextLoader(str(md_file), encoding='utf-8')
                docs = loader.load()
                documents.extend(docs)
                loaded_files.append(md_file.name)
                print(f"✓ Loaded: {md_file.name}")
            except Exception as e:
                errors.append(f"{md_file.name}: {e}")
                print(f"✗ Error loading {md_file.name}: {e}")
        
        # Check if any documents were loaded
        if not documents:
            raise DocumentNotFoundError(
                "No documents found to ingest.\n"
                f"Please add .txt or .md files to: {docs_path}\n"
                f"Errors encountered: {errors if errors else 'None'}"
            )
        
        # Split documents into chunks
        # This enables more precise retrieval of relevant passages
        chunks = self.text_splitter.split_documents(documents)
        print(f"\nCreated {len(chunks)} chunks from {len(documents)} documents.")
        
        try:
            # Create vector store with embeddings
            # This stores both the text chunks and their embedding vectors
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.config.CHROMA_PERSIST_DIR,
                collection_name=self.config.COLLECTION_NAME
            )
            
            print(f"✓ Successfully ingested {len(chunks)} document chunks.")
            print(f"✓ Files loaded: {', '.join(loaded_files)}")
            return len(chunks)
            
        except Exception as e:
            raise DocumentIngestionError(
                f"Failed to create vector store: {e}\n"
                "Please check that ChromaDB is properly installed."
            )
    
    def add_document(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Add a single document to the vector store.
        
        Useful for adding custom content through the UI without
        requiring file uploads.
        
        Args:
            text: Document text content to add
            metadata: Optional metadata (source, date, etc.)
            
        Returns:
            Number of chunks created from the document
        """
        if not text or not text.strip():
            print("Warning: Empty document provided, nothing to add.")
            return 0
        
        # Initialize vector store if needed
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory=self.config.CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=self.config.COLLECTION_NAME
            )
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = [
            Document(
                page_content=chunk, 
                metadata=metadata or {"source": "user_input"}
            )
            for chunk in chunks
        ]
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        print(f"✓ Added document with {len(chunks)} chunks.")
        
        return len(chunks)
    
    # =========================================================================
    # QUERY-TIME RETRIEVAL PHASE
    # =========================================================================
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        This is the RETRIEVAL phase of RAG:
        1. Convert user query to embedding vector
        2. Search ChromaDB for similar document chunks
        3. Return top-K most relevant chunks with similarity scores
        
        HOW SIMILARITY SEARCH WORKS:
        - Query embedding is compared to all stored embeddings
        - Cosine similarity measures how "close" vectors are
        - Higher score = more relevant to the query
        
        DEFENSIVE DESIGN:
        - Validates query is non-empty and within length limits
        - Gracefully handles empty knowledge base
        - Catches and logs retrieval errors without crashing
        
        Args:
            query: User's question to search for
            top_k: Number of results to return (default: 3)
            
        Returns:
            List of (Document, score) tuples, sorted by relevance
        """
        # DEFENSIVE: Handle empty knowledge base gracefully
        if not self.vector_store:
            print("Warning: Knowledge base is empty. Please ingest documents first.")
            return []
        
        # DEFENSIVE: Validate query input
        if not query or not query.strip():
            print("Warning: Empty query provided.")
            return []
        
        # DEFENSIVE: Truncate excessively long queries
        query = query.strip()
        if len(query) > MAX_QUERY_LENGTH:
            print(f"Warning: Query truncated from {len(query)} to {MAX_QUERY_LENGTH} chars.")
            query = query[:MAX_QUERY_LENGTH]
        
        # DEFENSIVE: Validate top_k parameter
        top_k = top_k or self.config.TOP_K_RESULTS
        top_k = max(1, min(top_k, 10))  # Clamp between 1 and 10
        
        try:
            # Perform similarity search with relevance scores
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get formatted context string for a query.
        
        This method combines retrieval with formatting:
        1. Retrieve relevant document chunks
        2. Format them into a single context string
        3. Include source information for transparency
        
        The formatted context is passed to the LLM for grounding.
        
        OUTPUT FORMAT (Interview Talking Point):
        ```
        [Document 1] (Source: faq.txt, Relevance: 0.85)
        <chunk content>
        
        ---
        
        [Document 2] (Source: policy.txt, Relevance: 0.72)
        <chunk content>
        ```
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context string for LLM prompt, or error message
        """
        results = self.retrieve(query, top_k)
        
        # DEFENSIVE: Handle case where no documents are retrieved
        if not results:
            return NO_CONTEXT_MESSAGE
        
        # Format results into a readable context string
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            # Extract source file name from metadata (defensive path handling)
            source = doc.metadata.get('source', 'Unknown')
            try:
                source = Path(source).name if source != 'Unknown' else source
            except (ValueError, OSError):
                source = 'Unknown'
            
            # Format relevance score for transparency (0-1 scale, inverted from distance)
            relevance = max(0.0, min(1.0, 1.0 - score)) if score else 0.0
            
            # Format each document chunk with its source and relevance
            context_parts.append(
                f"[Document {i}] (Source: {source}, Relevance: {relevance:.2f})\n{doc.page_content}"
            )
        
        # Join all chunks with separators
        return "\n\n---\n\n".join(context_parts)
    
    def is_ready(self) -> bool:
        """
        Check if the retriever has documents and is ready to use.
        
        Returns:
            True if knowledge base has documents, False otherwise
        """
        return self.vector_store is not None
    
    def clear_vector_store(self) -> None:
        """
        Clear all documents from the vector store.
        
        Use this to reset the knowledge base and start fresh.
        """
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                self.vector_store = None
                print("✓ Vector store cleared successfully.")
            except Exception as e:
                print(f"Error clearing vector store: {e}")


# =============================================================================
# SINGLETON PATTERN FOR EFFICIENCY
# =============================================================================

_retriever_instance: Optional[DocumentRetriever] = None


def get_retriever() -> DocumentRetriever:
    """
    Get or create the singleton retriever instance.
    
    Using a singleton ensures:
    - Embedding model is only loaded once
    - Vector store connection is reused
    - Memory efficient for the Streamlit app
    
    Returns:
        The shared DocumentRetriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DocumentRetriever()
    return _retriever_instance