"""
Main document service that orchestrates PDF processing, vector storage, and chat functionality.
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from .pdf_processor import PDFProcessor
from .vector_service import VectorService
from .chat_service import ChatService
from ..models import FileInfo, ProcessingResult, DocumentChunk
from ..utils import (
    generate_session_id,
    generate_index_name,
    validate_file_type,
    validate_file_size,
    measure_time,
    log_processing_info,
    handle_processing_error
)
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """Main service for document processing and chat functionality."""
    
    def __init__(self):
        """Initialize the document service."""
        self.pdf_processor = PDFProcessor()
        self.vector_service = VectorService()
        self.chat_service = ChatService()
    
    def _validate_files(self, file_contents: List[Dict[str, Any]]) -> List[FileInfo]:
        """
        Validate uploaded files.
        
        Args:
            file_contents: List of file information dictionaries
            
        Returns:
            List of validated FileInfo objects
            
        Raises:
            ValueError: If validation fails
        """
        validated_files = []
        
        for file_info in file_contents:
            filename = file_info['filename']
            content = file_info['content']
            
            # Validate file type
            if not validate_file_type(filename):
                raise ValueError(f"File type not allowed: {filename}")
            
            # Validate file size using utility function first
            if not validate_file_size(len(content)):
                raise ValueError(f"File {filename} is too large: {len(content)/1024/1024:.1f}MB exceeds maximum allowed size")
            
            try:
                # Create FileInfo object (this will trigger Pydantic validation)
                file_obj = FileInfo(
                    filename=filename,
                    content=content,
                    size=len(content)
                )
                
                validated_files.append(file_obj)
                
            except Exception as e:
                # Handle Pydantic validation errors
                if "File size" in str(e):
                    raise ValueError(f"File {filename} is too large: {len(content)/1024/1024:.1f}MB exceeds maximum allowed size")
                else:
                    raise ValueError(f"File validation failed for {filename}: {str(e)}")
        
        return validated_files
    
    @measure_time
    def process_documents(
        self, 
        file_contents: List[Dict[str, Any]], 
        session_id: Optional[str] = None
    ) -> ProcessingResult:
        """
        Process uploaded documents and store them in the vector database.
        
        Args:
            file_contents: List of file information dictionaries
            session_id: Optional session ID, will be generated if not provided
            
        Returns:
            ProcessingResult with processing information
        """
        try:
            collection_name = generate_index_name(session_id)
            print(f"Collection name: {collection_name}")
            log_processing_info("Document processing started", {
                "session_id": session_id,
                "collection_name": collection_name,
                "files_count": len(file_contents)
            })
            
            # Validate files
            validated_files = self._validate_files(file_contents)
            
            # Convert to format expected by PDF processor
            pdf_files = [
                {
                    'filename': file.filename,
                    'content': file.content
                }
                for file in validated_files
            ]
            
            # Extract text from PDFs
            documents = self.pdf_processor.process_multiple_pdfs(pdf_files)
            if not documents:
                raise Exception("No text could be extracted from the uploaded files")
            
            # Split documents into chunks
            chunks = self.pdf_processor.split_documents_into_chunks(documents)
            print(f"Chunks: {chunks}")
            # Store in vector database
            self.vector_service.store_documents(chunks, collection_name)
            print(f"Chunks stored successfully")
            result = ProcessingResult(
                status="success",
                files_processed=len(validated_files),
                chunks_created=len(chunks),
                index_name=collection_name
            )
            log_processing_info("Document processing completed", {
                "session_id": session_id,
                "collection_name": collection_name,
                "files_processed": result.files_processed,
                "chunks_created": result.chunks_created
            })
            
            return result
            
        except Exception as e:
            error_info = handle_processing_error(
                "document_processing",
                e,
                {
                    "session_id": session_id,
                    "files_count": len(file_contents)
                }
            )
            
            return ProcessingResult(
                status="error",
                files_processed=0,
                chunks_created=0,
                index_name=collection_name if 'collection_name' in locals() else "",
                error=str(e)
            )
    
    @measure_time
    def chat_with_documents(
        self, 
        question: str, 
        session_id: str,
        index_name: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, List[str], float]:
        """
        Chat with documents using the vector database and LLM.
        
        Args:
            question: User's question
            session_id: Session ID to identify the document collection
            max_tokens: Maximum tokens for the response
            
        Returns:
            Tuple of (answer, sources, confidence)
        """
        try:
            # Use provided index_name or generate from session_id
            collection_name = index_name if index_name else generate_index_name(session_id)
            
            # Check if session has documents first
            if not self.session_has_documents(session_id, collection_name):
                raise Exception("No documents found for this session. Please upload PDF documents first before asking questions.")
            
            log_processing_info("Chat query started", {
                "session_id": session_id,
                "collection_name": collection_name,
                "question_length": len(question)
            })
            
            # Search for relevant documents
            relevant_docs = self.vector_service.search_similar_documents(
                query=question,
                collection_name=collection_name
            )
            
            # Generate response
            answer, sources, confidence = self.chat_service.generate_response(
                question=question,
                documents=relevant_docs,
                max_tokens=max_tokens
            )
            
            log_processing_info("Chat query completed", {
                "session_id": session_id,
                "answer_length": len(answer),
                "sources_count": len(sources),
                "confidence": confidence
            })
            
            return answer, sources, confidence
            
        except Exception as e:
            error_info = handle_processing_error(
                "chat_query",
                e,
                {
                    "session_id": session_id,
                    "question": question
                }
            )
            
            # Provide user-friendly error messages
            error_message = str(e)
            if "does not exist" in error_message:
                raise Exception(f"No documents found for this session. Please upload PDF documents first before asking questions.")
            elif "Collection" in error_message and "doesn't exist" in error_message:
                raise Exception(f"Session not found or no documents uploaded. Please upload PDF documents first.")
            else:
                raise Exception(f"Failed to process chat query: {error_info}")
    
    def create_session(self) -> Tuple[str, str]:
        """
        Create a new session.
        
        Returns:
            Tuple of (session_id, collection_name)
        """
        session_id = generate_session_id()
        collection_name = generate_index_name(session_id)
        
        log_processing_info("Session created", {
            "session_id": session_id,
            "collection_name": collection_name
        })
        
        return session_id, collection_name
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its associated data.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            collection_name = generate_index_name(session_id)
            success = self.vector_service.delete_collection(collection_name)
            
            log_processing_info("Session deleted", {
                "session_id": session_id,
                "collection_name": collection_name,
                "success": success
            })
            
            return success
            
        except Exception as e:
            error_info = handle_processing_error(
                "session_deletion",
                e,
                {"session_id": session_id}
            )
            logger.error(f"Failed to delete session: {error_info}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with session information or None if not found
        """
        try:
            collection_name = generate_index_name(session_id)
            collection_info = self.vector_service.get_collection_info(collection_name)
            
            if collection_info:
                collection_info['session_id'] = session_id
                collection_info['collection_name'] = collection_name
            
            return collection_info
            
        except Exception as e:
            error_info = handle_processing_error(
                "session_info",
                e,
                {"session_id": session_id}
            )
            logger.error(f"Failed to get session info: {error_info}")
            return None
    
    def session_has_documents(self, session_id: str, index_name: Optional[str] = None) -> bool:
        """
        Check if a session has documents uploaded.
        
        Args:
            session_id: Session ID
            index_name: Optional collection name (will be generated from session_id if not provided)
            
        Returns:
            True if session has documents, False otherwise
        """
        try:
            collection_name = index_name if index_name else generate_index_name(session_id)
            collection_info = self.vector_service.get_collection_info(collection_name)
            return collection_info is not None and collection_info.get('points_count', 0) > 0
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dictionary with health status information
        """
        try:
            vector_health = self.vector_service.health_check()
            chat_health = self.chat_service.health_check()
            
            overall_status = "healthy" if (
                vector_health.get("status") == "healthy" and 
                chat_health.get("status") == "healthy"
            ) else "unhealthy"
            
            return {
                "status": overall_status,
                "vector_service": vector_health,
                "chat_service": chat_health,
                "timestamp": log_processing_info("Health check", {})[0] if hasattr(log_processing_info, '__call__') else None
            }
            
        except Exception as e:
            error_info = handle_processing_error("health_check", e)
            return {
                "status": "unhealthy",
                "error": str(e)
            }
