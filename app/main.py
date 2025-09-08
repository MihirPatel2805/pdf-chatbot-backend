"""
FastAPI application for PDF Chatbot Backend.
"""

import time
from typing import List
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .config import settings, validate_required_settings
from .models import (
    UploadRequest, UploadResponse, ChatRequest, ChatResponse,
    SessionCreateResponse, HealthResponse, ErrorResponse,
    UserSessionCreateRequest, UserSessionResponse, UserSessionsListResponse,
    UserProfileResponse, UserChatRequest, UserUploadRequest
)
from .services import DocumentService
from .services.user_session_service import UserSessionService
from .db import get_db
from sqlalchemy.orm import Session as OrmSession
from .models_db import Base
from .db import engine
from .services.db_repositories import ChatMessageRepository
from .auth import get_current_user, ClerkUser
from .utils import format_timestamp, generate_index_name, generate_session_id

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate required settings on startup
try:
    validate_required_settings()
except ValueError as e:
    logger.error(f"Configuration validation failed: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A secure PDF chatbot backend with in-memory processing",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_service = DocumentService()
user_session_service = UserSessionService(document_service)
chat_message_repo = ChatMessageRepository()


@app.on_event("startup")
def on_startup_create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ensured.")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else "An unexpected error occurred",
            status_code=500
        ).dict()
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "PDF Chatbot API is running",
        "version": settings.app_version,
        "timestamp": format_timestamp()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_info = document_service.health_check()
        
        return HealthResponse(
            status=health_info.get("status", "unknown"),
            message="Service health check completed",
            version=settings.app_version,
            timestamp=format_timestamp()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.post("/create-session", response_model=SessionCreateResponse)
async def create_session():
    """Create a new chatbot session."""
    try:
        session_id, index_name = document_service.create_session()
        
        return SessionCreateResponse(
            session_id=session_id,
            index_name=index_name,
            message="New session created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to create session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    name: str = Form(..., min_length=1, max_length=100),
    description: str = Form(..., min_length=1, max_length=500),
    session_id: str = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    Upload PDF files for processing.
    
    This endpoint processes PDFs directly from memory without storing them on disk.
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 10:  # Limit number of files
            raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files allowed.")
        
        # Read file contents into memory
        file_contents = []
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type: {file.filename}. Only PDF files are allowed."
                )
            
            # Read file content
            content = await file.read()
            
            # Validate file size
            file_size_mb = len(content) / (1024 * 1024)
            if len(content) > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large: {file_size_mb:.1f}MB. Maximum size is {settings.max_file_size_mb}MB."
                )
            
            file_contents.append({
                'filename': file.filename,
                'content': content
            })
        
        # Process documents
        start_time = time.time()
        result = document_service.process_documents(file_contents, session_id)
        processing_time = time.time() - start_time
        
        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error)
    
        return UploadResponse(
            message="PDFs uploaded and processed successfully",
            session_id=result.index_name.split('-')[-1],  # Extract session ID from index name
            index_name=result.index_name,
            files_processed=result.files_processed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded files")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """Chat with uploaded PDF documents."""
    try:
        start_time = time.time()
        
        # Process chat query
        answer, sources, confidence = document_service.chat_with_documents(
            question=request.question,
            session_id=request.session_id,
            index_name=request.index_name,
            max_tokens=request.max_tokens
        )
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat query")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data."""
    try:
        success = document_service.delete_session(session_id)
        
        if success:
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a session."""
    try:
        session_info = document_service.get_session_info(session_id)
        
        if session_info:
            return session_info
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get session information")


# ============================================================================
# AUTHENTICATED USER ENDPOINTS
# ============================================================================

@app.get("/user/profile", response_model=UserProfileResponse)
async def get_user_profile(current_user: ClerkUser = Depends(get_current_user), db: OrmSession = Depends(get_db)):
    """Get user profile with session statistics."""
    try:
        profile = user_session_service.get_user_profile(db, current_user.user_id)
        
        # Update with user information from Clerk
        profile.email = current_user.email
        profile.first_name = current_user.first_name
        profile.last_name = current_user.last_name
        profile.full_name = current_user.full_name
        
        return profile
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user profile")


@app.post("/user/sessions", response_model=UserSessionResponse)
async def create_user_session(
    request: UserSessionCreateRequest = Body(..., description="Session create payload"),
    current_user: ClerkUser = Depends(get_current_user),
    db: OrmSession = Depends(get_db)
):
    """Create a new session for the authenticated user."""
    try:
        print('creating user session')
        print('current_user', current_user)
        print('request', request)
        session_id = generate_session_id()
        index_name = generate_index_name(session_id)
        session = user_session_service.create_user_session(
            db=db,
            user_id=current_user.user_id,
            name=request.name,
            description=request.description,
            session_id=session_id,
            index_name=index_name
        )
        
        return session
        
    except Exception as e:
        logger.error(f"Failed to create user session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.get("/user/sessions", response_model=UserSessionsListResponse)
async def get_user_sessions(current_user: ClerkUser = Depends(get_current_user), db: OrmSession = Depends(get_db)):
    """Get all sessions for the authenticated user."""
    try:
        sessions = user_session_service.get_user_sessions(db, current_user.user_id)
        
        return UserSessionsListResponse(
            sessions=sessions,
            total_count=len(sessions),
            user_id=current_user.user_id
        )
        
    except Exception as e:
        logger.error(f"Failed to get user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user sessions")


@app.get("/user/sessions/{session_id}", response_model=UserSessionResponse)
async def get_user_session(
    session_id: str,
    current_user: ClerkUser = Depends(get_current_user),
    db: OrmSession = Depends(get_db)
):
    """Get a specific session for the authenticated user."""
    try:
        # Validate user access to session
        if not user_session_service.validate_user_session_access(db, current_user.user_id, session_id):
            raise HTTPException(status_code=404, detail="Session not found or access denied")
        
        session = user_session_service.get_user_session(db, current_user.user_id, session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get user session")


@app.delete("/user/sessions/{session_id}")
async def delete_user_session(
    session_id: str,
    current_user: ClerkUser = Depends(get_current_user),
    db: OrmSession = Depends(get_db)
):
    """Delete a session for the authenticated user."""
    try:
        # Validate user access to session
        if not user_session_service.validate_user_session_access(db, current_user.user_id, session_id):
            raise HTTPException(status_code=404, detail="Session not found or access denied")
        
        success = user_session_service.delete_user_session(db, current_user.user_id, session_id)
        
        if success:
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user session: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@app.post("/user/upload-pdf", response_model=UploadResponse)
async def upload_pdf_user(
    name: str = Form(..., min_length=1, max_length=100),
    description: str = Form(..., min_length=1, max_length=500),
    session_id: str = Form(None),
    files: List[UploadFile] = File(...),
    current_user: ClerkUser = Depends(get_current_user),
    db: OrmSession = Depends(get_db)
):
    """
    Upload PDF files for processing (authenticated user version).
    
    This endpoint processes PDFs directly from memory without storing them on disk.
    """
    try:
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 10:  # Limit number of files
            raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files allowed.")
        
        # If session_id provided, validate user access
        if session_id and not user_session_service.validate_user_session_access(db, current_user.user_id, session_id):
            raise HTTPException(status_code=403, detail="Access denied to specified session")
        
        # Read file contents into memory
        file_contents = []
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid file type: {file.filename}. Only PDF files are allowed."
                )
            
            # Read file content
            content = await file.read()
            
            # Validate file size
            file_size_mb = len(content) / (1024 * 1024)
            if len(content) > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large: {file_size_mb:.1f}MB. Maximum size is {settings.max_file_size_mb}MB."
                )
            
            file_contents.append({
                'filename': file.filename,
                'content': content
            })
        # Generate session ID if not provided
        if not session_id:
            session_id = generate_session_id()
            
        # Process documents
        start_time = time.time()
        result = document_service.process_documents(file_contents, session_id)
        processing_time = time.time() - start_time
        print('result', result)
        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error)
        #update the session in the database if not present
        get_session = user_session_service.get_user_session(db, current_user.user_id, session_id)
        print('get_session', get_session)
        if not get_session:
            user_session_service.create_user_session(db, current_user.user_id, name, description, session_id, generate_index_name(session_id))
            user_session_service.update_session_document_count(db, session_id, result.files_processed)
        else:
            user_session_service.update_session_document_count(db, session_id, result.files_processed)

        return UploadResponse(
            message="PDFs uploaded and processed successfully",
            session_id=session_id,  # Extract session ID from index name
            index_name=generate_index_name(session_id),
            files_processed=result.files_processed
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process uploaded files")


@app.post("/user/chat", response_model=ChatResponse)
async def chat_with_pdf_user(
    request: UserChatRequest = Body(..., description="Chat request payload"),
    current_user: ClerkUser = Depends(get_current_user),
    db: OrmSession = Depends(get_db)
):
    """Chat with uploaded PDF documents (authenticated user version)."""
    try:
        # Validate user access to session
        if not user_session_service.validate_user_session_access(db, current_user.user_id, request.session_id):
            raise HTTPException(status_code=403, detail="Access denied to specified session")
        
        start_time = time.time()
        
        # Process chat query
        answer, sources, confidence = document_service.chat_with_documents(
            question=request.question,
            session_id=request.session_id,
            index_name=None,  # Will be generated from session_id
            max_tokens=request.max_tokens
        )
        
        processing_time = time.time() - start_time
        
        # Persist chat messages
        try:
            chat_message_repo.add_message(db, session_id=request.session_id, role="user", content=request.question)
            chat_message_repo.add_message(db, session_id=request.session_id, role="assistant", content=answer, sources="\n".join(sources) if sources else None, confidence=confidence)
        except Exception as e:
            logger.warning(f"Failed to persist chat messages: {e}")

        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=request.session_id,
            confidence=confidence,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat query failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process chat query")


# Legacy endpoint for backward compatibility
@app.post("/upload-pdf-legacy", response_model=UploadResponse)
async def upload_pdf_legacy(
    name: str = Form(...),
    description: str = Form(...),
    session_id: str = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    Legacy upload endpoint that stores files on disk.
    
    DEPRECATED: Use /user/upload-pdf instead for better security and performance.
    """
    logger.warning("Legacy upload endpoint used. Consider migrating to /user/upload-pdf")
    
    # For now, redirect to the new endpoint
    return await upload_pdf(name, description, session_id, files)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )