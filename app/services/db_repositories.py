from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select

from ..models_db import User, Session as DbSession, ChatMessage


class UserRepository:
    def get_or_create(self, db: Session, user_id: str, **kwargs) -> User:
        user = db.get(User, user_id)
        if user:
            return user
        user = User(id=user_id, **kwargs)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


class SessionRepository:
    def create(self, db: Session, *, id: str, user_id: str, index_name: str, name: Optional[str], description: Optional[str]) -> DbSession:
        sess = DbSession(id=id, user_id=user_id, index_name=index_name, name=name, description=description)
        db.add(sess)
        db.commit()
        db.refresh(sess)
        return sess

    def list_for_user(self, db: Session, user_id: str) -> List[DbSession]:
        return db.scalars(select(DbSession).where(DbSession.user_id == user_id).order_by(DbSession.created_at.desc())).all()

    def get_for_user(self, db: Session, user_id: str, session_id: str) -> Optional[DbSession]:
        sess = db.get(DbSession, session_id)
        if not sess or sess.user_id != user_id:
            return None
        return sess

    def delete_for_user(self, db: Session, user_id: str, session_id: str) -> bool:
        sess = self.get_for_user(db, user_id, session_id)
        if not sess:
            return False
        db.delete(sess)
        db.commit()
        return True

    def update_document_count(self, db: Session, session_id: str, document_count: int) -> bool:
        sess = db.get(DbSession, session_id)
        if not sess:
            return False
        sess.document_count = sess.document_count + document_count
        db.commit()
        return True


class ChatMessageRepository:
    def add_message(self, db: Session, *, session_id: str, role: str, content: str, sources: Optional[str] = None, confidence: Optional[float] = None) -> ChatMessage:
        msg = ChatMessage(session_id=session_id, role=role, content=content, sources=sources, confidence=confidence)
        db.add(msg)
        db.commit()
        db.refresh(msg)
        return msg

    def list_for_session(self, db: Session, session_id: str) -> List[ChatMessage]:
        return db.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc())).all()

