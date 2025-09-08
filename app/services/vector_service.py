"""
Vector database service for managing embeddings and similarity search.
"""

from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from ..config import settings
from ..utils import (
    measure_time,
    log_processing_info,
    handle_processing_error
)
import logging

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector database operations."""
    
    def __init__(self):
        """Initialize the vector service."""
        self.client = self._initialize_qdrant_client()
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_qdrant_client(self) -> QdrantClient:
        """Initialize Qdrant client."""
        try:
            if settings.qdrant_api_key:
                client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key
                )
            else:
                client = QdrantClient(url=settings.qdrant_url)
            
            log_processing_info("Qdrant client initialized", {
                "url": settings.qdrant_url,
                "has_api_key": bool(settings.qdrant_api_key)
            })
            
            return client
            
        except Exception as e:
            error_info = handle_processing_error("qdrant_client_init", e)
            raise Exception(f"Failed to initialize Qdrant client: {error_info}")
    
    def _initialize_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Initialize Google Generative AI embeddings."""
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.google_embedding_model,
                api_key=settings.google_api_key
            )
            
            log_processing_info("Embeddings initialized", {
                "model": settings.google_embedding_model
            })
            
            return embeddings
            
        except Exception as e:
            error_info = handle_processing_error("embeddings_init", e)
            raise Exception(f"Failed to initialize embeddings: {error_info}")
    
    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection to create
            
        Returns:
            True if created successfully, False if already exists
        """
        try:
            # Check if collection already exists
            try:
                self.client.get_collection(collection_name)
                log_processing_info("Collection already exists", {
                    "collection_name": collection_name
                })
                return False
                
            except Exception:
                # Collection doesn't exist, create it
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=settings.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                
                log_processing_info("Collection created", {
                    "collection_name": collection_name,
                    "vector_dimension": settings.vector_dimension
                })
                
                return True
                
        except Exception as e:
            error_info = handle_processing_error(
                "collection_creation",
                e,
                {"collection_name": collection_name}
            )
            raise Exception(f"Failed to create collection {collection_name}: {error_info}")
    
    @measure_time
    def store_documents(self, documents: List[Document], collection_name: str) -> bool:
        """
        Store documents in the vector database.
        
        Args:
            documents: List of Document objects to store
            collection_name: Name of the collection
            
        Returns:
            True if stored successfully
        """
        try:
            # Ensure collection exists
            self.create_collection(collection_name)
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings
            )
            
            # Generate unique IDs for documents
            from uuid import uuid4
            document_ids = [str(uuid4()) for _ in range(len(documents))]
            
            # Store documents
            vector_store.add_documents(documents=documents, ids=document_ids)
            
            log_processing_info("Documents stored successfully", {
                "collection_name": collection_name,
                "document_count": len(documents)
            })
            
            return True
            
        except Exception as e:
            error_info = handle_processing_error(
                "document_storage",
                e,
                {
                    "collection_name": collection_name,
                    "document_count": len(documents)
                }
            )
            raise Exception(f"Failed to store documents: {error_info}")
    
    @measure_time
    def search_similar_documents(
        self, 
        query: str, 
        collection_name: str, 
        k: int = None
    ) -> List[Document]:
        """
        Search for similar documents in the vector database.
        
        Args:
            query: Search query
            collection_name: Name of the collection to search
            k: Number of similar documents to return
            
        Returns:
            List of similar Document objects
        """
        if k is None:
            k = settings.similarity_search_k
        
        try:
            # Check if collection exists first
            try:
                self.client.get_collection(collection_name)
            except Exception:
                raise Exception(f"Collection '{collection_name}' does not exist. Please upload documents first.")
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings
            )
            
            # Search for similar documents
            similar_docs = vector_store.similarity_search(query, k=k)
            
            log_processing_info("Similarity search completed", {
                "collection_name": collection_name,
                "query_length": len(query),
                "results_count": len(similar_docs),
                "k": k
            })
            
            return similar_docs
            
        except Exception as e:
            error_info = handle_processing_error(
                "similarity_search",
                e,
                {
                    "collection_name": collection_name,
                    "query": query,
                    "k": k
                }
            )
            raise Exception(f"Failed to search similar documents: {error_info}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            
            log_processing_info("Collection deleted", {
                "collection_name": collection_name
            })
            
            return True
            
        except Exception as e:
            error_info = handle_processing_error(
                "collection_deletion",
                e,
                {"collection_name": collection_name}
            )
            logger.warning(f"Failed to delete collection {collection_name}: {error_info}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information or None if not found
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
            
        except Exception as e:
            error_info = handle_processing_error(
                "collection_info",
                e,
                {"collection_name": collection_name}
            )
            logger.warning(f"Failed to get collection info: {error_info}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector service.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Test client connection
            collections = self.client.get_collections()
            
            return {
                "status": "healthy",
                "qdrant_url": settings.qdrant_url,
                "collections_count": len(collections.collections),
                "embedding_model": settings.google_embedding_model
            }
            
        except Exception as e:
            error_info = handle_processing_error("health_check", e)
            return {
                "status": "unhealthy",
                "error": str(e),
                "qdrant_url": settings.qdrant_url
            }
