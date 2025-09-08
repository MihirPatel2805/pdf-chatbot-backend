"""
Pydantic models for request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import uuid
from datetime import datetime


class UploadRequest(BaseModel):
    """Request model for PDF upload."""
    name: str = Field(..., min_length=1, max_length=100, description="Name for the document collection")
    description: str = Field(..., min_length=1, max_length=500, description="Description of the documents")
    session_id: Optional[str] = Field(default=None, description="Optional session ID, will be generated if not provided")


class UploadResponse(BaseModel):
    """Response model for PDF upload."""
    message: str = Field(..., description="Success message")
    session_id: str = Field(..., description="Session ID for this upload")
    index_name: str = Field(..., description="Vector database collection name")
    files_processed: int = Field(..., description="Number of files processed")
    status: str = Field(default="success", description="Processing status")


class ChatRequest(BaseModel):
    """Request model for chat queries."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: str = Field(..., description="Session ID to identify the document collection")
    index_name: Optional[str] = Field(default=None, description="Vector database collection name (optional, will be generated from session_id if not provided)")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=2000, description="Maximum tokens for response")


class ChatResponse(BaseModel):
    """Response model for chat queries."""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    session_id: str = Field(..., description="Session ID")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


class SessionCreateResponse(BaseModel):
    """Response model for session creation."""
    session_id: str = Field(..., description="Generated session ID")
    index_name: str = Field(..., description="Vector database collection name")
    message: str = Field(..., description="Success message")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class FileInfo(BaseModel):
    """Model for file information."""
    filename: str = Field(..., description="Name of the file")
    content: bytes = Field(..., description="File content as bytes")
    size: int = Field(..., description="File size in bytes")
    
    @validator('size')
    def validate_file_size(cls, v, values):
        """Validate file size."""
        # Import here to avoid circular imports
        from .config import settings
        max_size = settings.max_file_size_mb * 1024 * 1024
        if v > max_size:
            raise ValueError(f"File size {v} bytes ({v/1024/1024:.1f}MB) exceeds maximum allowed size {max_size} bytes ({settings.max_file_size_mb}MB)")
        return v


class DocumentChunk(BaseModel):
    """Model for document chunks."""
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique chunk ID")


class ProcessingResult(BaseModel):
    """Model for processing results."""
    status: str = Field(..., description="Processing status")
    files_processed: int = Field(..., description="Number of files processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    index_name: str = Field(..., description="Vector database collection name")
    error: Optional[str] = Field(default=None, description="Error message if any")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")


# User-related models
class UserInfo(BaseModel):
    """Model for user information."""
    user_id: str = Field(..., description="Clerk user ID")
    email: Optional[str] = Field(default=None, description="User email")
    first_name: Optional[str] = Field(default=None, description="User first name")
    last_name: Optional[str] = Field(default=None, description="User last name")
    full_name: str = Field(..., description="User full name")
    created_at: Optional[datetime] = Field(default=None, description="User creation timestamp")


class UserSession(BaseModel):
    """Model for user session information."""
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID who owns this session")
    index_name: str = Field(..., description="Vector database collection name")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access timestamp")
    document_count: int = Field(default=0, description="Number of documents in session")
    is_active: bool = Field(default=True, description="Whether session is active")


class UserSessionCreateRequest(BaseModel):
    """Request model for creating a user session."""
    name: Optional[str] = Field(default=None, max_length=100, description="Optional session name")
    description: Optional[str] = Field(default=None, max_length=500, description="Optional session description")


class UserSessionResponse(BaseModel):
    """Response model for user session operations."""
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    index_name: str = Field(..., description="Vector database collection name")
    created_at: datetime = Field(..., description="Session creation timestamp")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access timestamp")
    document_count: int = Field(default=0, description="Number of documents in session")
    is_active: bool = Field(default=True, description="Whether session is active")
    message: Optional[str] = Field(default=None, description="Response message")


class UserSessionsListResponse(BaseModel):
    """Response model for listing user sessions."""
    sessions: List[UserSessionResponse] = Field(..., description="List of user sessions")
    total_count: int = Field(..., description="Total number of sessions")
    user_id: str = Field(..., description="User ID")


class UserProfileResponse(BaseModel):
    """Response model for user profile."""
    user_id: str = Field(..., description="Clerk user ID")
    email: Optional[str] = Field(default=None, description="User email")
    first_name: Optional[str] = Field(default=None, description="User first name")
    last_name: Optional[str] = Field(default=None, description="User last name")
    full_name: str = Field(..., description="User full name")
    total_sessions: int = Field(default=0, description="Total number of sessions")
    active_sessions: int = Field(default=0, description="Number of active sessions")
    total_documents: int = Field(default=0, description="Total number of documents across all sessions")


# Updated request models with user context
class UserChatRequest(BaseModel):
    """Request model for chat queries with user context."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: str = Field(..., description="User session ID")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=2000, description="Maximum tokens for response")


class UserUploadRequest(BaseModel):
    """Request model for PDF upload with user context."""
    name: str = Field(..., min_length=1, max_length=100, description="Name for the document collection")
    description: str = Field(..., min_length=1, max_length=500, description="Description of the documents")
    session_id: Optional[str] = Field(default=None, description="Optional user session ID, will be generated if not provided")
