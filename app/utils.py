"""
Utility functions for the PDF Chatbot Backend.
"""

import time
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def generate_index_name(session_id: str) -> str:
    """Generate a collection name from session ID."""
    # Remove hyphens and take first 12 characters
    clean_id = session_id.replace('-', '')[:12]
    return f"{settings.qdrant_collection_prefix}-{clean_id}"


def validate_file_type(filename: str) -> bool:
    """Validate if the file type is allowed."""
    if not filename:
        return False
    
    file_extension = filename.lower().split('.')[-1]
    return file_extension in settings.allowed_file_types


def validate_file_size(file_size: int) -> bool:
    """Validate if the file size is within limits."""
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


def format_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat()


def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace dangerous characters
    dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    sanitized = filename
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        sanitized = name[:255-len(ext)-1] + ('.' + ext if ext else '')
    
    return sanitized


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """Extract metadata from filename."""
    return {
        'original_filename': filename,
        'sanitized_filename': sanitize_filename(filename),
        'file_extension': filename.lower().split('.')[-1] if '.' in filename else '',
        'upload_timestamp': format_timestamp()
    }


def create_document_metadata(filename: str, page_num: int, total_pages: int, 
                           additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create comprehensive metadata for a document."""
    base_metadata = {
        'source': filename,
        'filename': filename,
        'page': page_num,
        'total_pages': total_pages,
        'document_id': str(uuid.uuid4()),
        'created_at': format_timestamp()
    }
    
    if additional_metadata:
        base_metadata.update(additional_metadata)
    
    return base_metadata


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format."""
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False


def chunk_text_by_sentences(text: str, max_chunk_size: int = None) -> List[str]:
    """Split text into chunks by sentences."""
    if max_chunk_size is None:
        max_chunk_size = settings.chunk_size
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    import re
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    return text.strip()


def log_processing_info(operation: str, details: Dict[str, Any]) -> None:
    """Log processing information."""
    logger.info(f"{operation}: {details}")


def handle_processing_error(operation: str, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Handle and log processing errors."""
    error_info = {
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': format_timestamp()
    }
    
    if context:
        error_info.update(context)
    
    logger.error(f"Processing error: {error_info}")
    return error_info
