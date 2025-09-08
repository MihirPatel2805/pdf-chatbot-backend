"""
Chat service for generating responses using LLM.
"""

from typing import List, Dict, Any, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from ..config import settings
from ..utils import (
    measure_time,
    log_processing_info,
    handle_processing_error
)
import logging

logger = logging.getLogger(__name__)


class ChatService:
    """Service for generating chat responses using LLM."""
    
    def __init__(self):
        """Initialize the chat service."""
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the language model."""
        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.google_chat_model,
                api_key=settings.google_api_key,
                temperature=settings.google_temperature
            )
            
            log_processing_info("LLM initialized", {
                "model": settings.google_chat_model,
                "temperature": settings.google_temperature
            })
            
            return llm
            
        except Exception as e:
            error_info = handle_processing_error("llm_init", e)
            raise Exception(f"Failed to initialize LLM: {error_info}")
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for the LLM.
        
        Args:
            question: User's question
            context: Context from retrieved documents
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
You are a helpful AI assistant that answers questions based on the provided context from uploaded documents.

Instructions:
1. Answer the user's question based ONLY on the information provided in the context
2. If the context doesn't contain enough information to answer the question, clearly state this
3. Be concise but informative
4. If you reference specific information, mention the source document
5. If the question is not related to the document content, politely redirect the user

User Question: {question}

Context from documents:
{context}

Please provide a clear, accurate, and helpful answer based on the context provided.
"""
        return prompt.strip()
    
    def _extract_sources(self, documents: List[Document]) -> List[str]:
        """
        Extract source information from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of unique source filenames
        """
        sources = []
        for doc in documents:
            if hasattr(doc, 'metadata') and doc.metadata:
                source = doc.metadata.get('source', 'Unknown source')
                if source not in sources:
                    sources.append(source)
        return sources
    
    def _calculate_confidence(self, documents: List[Document], question: str) -> float:
        """
        Calculate confidence score based on document relevance.
        
        Args:
            documents: List of retrieved documents
            question: User's question
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not documents:
            return 0.0
        
        # Simple confidence calculation based on number of relevant documents
        # and their content length
        total_content_length = sum(len(doc.page_content) for doc in documents)
        avg_content_length = total_content_length / len(documents)
        
        # Normalize confidence (this is a simple heuristic)
        confidence = min(1.0, len(documents) * 0.2 + (avg_content_length / 1000) * 0.1)
        
        return round(confidence, 2)
    
    @measure_time
    def generate_response(
        self, 
        question: str, 
        documents: List[Document],
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[str], float]:
        """
        Generate a response to the user's question.
        
        Args:
            question: User's question
            documents: List of relevant documents
            max_tokens: Maximum tokens for the response
            
        Returns:
            Tuple of (answer, sources, confidence)
        """
        try:
            if not documents:
                return (
                    "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    [],
                    0.0
                )
            
            # Combine document content
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # Create prompt
            prompt = self._create_prompt(question, context)
            
            # Generate response
            response_kwargs = {}
            if max_tokens:
                response_kwargs['max_tokens'] = max_tokens
            else:
                response_kwargs['max_tokens'] = settings.google_max_tokens
            
            response = self.llm.invoke(prompt, **response_kwargs)
            answer = response.content
            
            # Extract sources and calculate confidence
            sources = self._extract_sources(documents)
            confidence = self._calculate_confidence(documents, question)
            
            log_processing_info("Response generated", {
                "question_length": len(question),
                "context_length": len(context),
                "answer_length": len(answer),
                "sources_count": len(sources),
                "confidence": confidence
            })
            
            return answer, sources, confidence
            
        except Exception as e:
            error_info = handle_processing_error(
                "response_generation",
                e,
                {
                    "question": question,
                    "documents_count": len(documents)
                }
            )
            raise Exception(f"Failed to generate response: {error_info}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the chat service.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test LLM with a simple prompt
            test_response = self.llm.invoke("Hello, are you working?")
            
            return {
                "status": "healthy",
                "model": settings.google_chat_model,
                "test_response_length": len(test_response.content)
            }
            
        except Exception as e:
            error_info = handle_processing_error("chat_health_check", e)
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": settings.google_chat_model
            }
