"""
User session management service (DB-backed) for handling user-specific sessions.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from sqlalchemy.orm import Session as OrmSession

from ..models import UserSessionResponse, UserProfileResponse
from ..utils import generate_session_id, generate_index_name
from .document_service import DocumentService
from .db_repositories import UserRepository, SessionRepository

logger = logging.getLogger(__name__)


class UserSessionService:
    """Service for managing user-specific sessions using the database."""
    
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
        self.users = UserRepository()
        self.sessions = SessionRepository()
    
    def create_user_session(self, db: OrmSession, user_id: str, name: Optional[str] = None, 
                            description: Optional[str] = None,session_id:str = None,index_name:str = None) -> UserSessionResponse:
        try:
            # Ensure user exists
            self.users.get_or_create(db, user_id)


            sess = self.sessions.create(
                db,
                id=session_id,
                user_id=user_id,
                index_name=index_name,
                name=name,
                description=description
            )

            return UserSessionResponse(
                session_id=sess.id,
                user_id=sess.user_id,
                index_name=sess.index_name,
                created_at=sess.created_at,
                last_accessed=sess.last_accessed,
                document_count=sess.document_count,
                is_active=sess.is_active,
                message="Session created successfully"
            )
        except Exception as e:
            logger.error(f"Failed to create user session: {str(e)}")
            raise Exception(f"Failed to create session: {str(e)}")

    def get_user_sessions(self, db: OrmSession, user_id: str) -> List[UserSessionResponse]:
        try:
            rows = self.sessions.list_for_user(db, user_id)
            responses: List[UserSessionResponse] = []
            for s in rows:
                # optional: recompute document_count from vector store
                try:
                    info = self.document_service.get_session_info(s.id)
                    doc_count = info.get('points_count', 0) if info else 0
                except Exception:
                    doc_count = s.document_count
                responses.append(UserSessionResponse(
                    session_id=s.id,
                    user_id=s.user_id,
                    index_name=s.index_name,
                    created_at=s.created_at,
                    last_accessed=s.last_accessed,
                    document_count=doc_count,
                    is_active=s.is_active
                ))
            return responses
        except Exception as e:
            logger.error(f"Failed to get user sessions: {str(e)}")
            raise Exception(f"Failed to get user sessions: {str(e)}")

    def get_user_session(self, db: OrmSession, user_id: str, session_id: str) -> Optional[UserSessionResponse]:
        try:
            s = self.sessions.get_for_user(db, user_id, session_id)
            if not s:
                return None
            try:
                info = self.document_service.get_session_info(session_id)
                doc_count = info.get('points_count', 0) if info else s.document_count
            except Exception:
                doc_count = s.document_count
            return UserSessionResponse(
                session_id=s.id,
                user_id=s.user_id,
                index_name=s.index_name,
                created_at=s.created_at,
                last_accessed=s.last_accessed,
                document_count=doc_count,
                is_active=s.is_active
            )
        except Exception as e:
            logger.error(f"Failed to get user session: {str(e)}")
            return None

    def delete_user_session(self, db: OrmSession, user_id: str, session_id: str) -> bool:
        try:
            # Delete from vector store first
            success = self.document_service.delete_session(session_id)
            if not success:
                return False
            # Delete from DB
            return self.sessions.delete_for_user(db, user_id, session_id)
        except Exception as e:
            logger.error(f"Failed to delete user session: {str(e)}")
            return False

    def validate_user_session_access(self, db: OrmSession, user_id: str, session_id: str) -> bool:
        try:
            s = self.sessions.get_for_user(db, user_id, session_id)
            return s is not None and s.is_active
        except Exception as e:
            logger.error(f"Failed to validate user session access: {str(e)}")
            return False

    def get_user_profile(self, db: OrmSession, user_id: str) -> UserProfileResponse:
        try:
            sessions = self.get_user_sessions(db, user_id)
            total_sessions = len(sessions)
            active_sessions = len([s for s in sessions if s.is_active])
            total_documents = sum(s.document_count for s in sessions)
            return UserProfileResponse(
                user_id=user_id,
                total_sessions=total_sessions,
                active_sessions=active_sessions,
                total_documents=total_documents,
                full_name="User"
            )
        except Exception as e:
            logger.error(f"Failed to get user profile: {str(e)}")
            raise Exception(f"Failed to get user profile: {str(e)}")

    def update_session_document_count(self, db: OrmSession, session_id: str, document_count: int) -> bool:
        try:
            return self.sessions.update_document_count(db, session_id, document_count)
        except Exception as e:
            logger.error(f"Failed to update session document count: {str(e)}")
            return False
