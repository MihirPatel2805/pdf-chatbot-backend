from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


# Normalize DATABASE_URL for SQLAlchemy if needed
def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        # Convert deprecated postgres:// to postgresql+psycopg2://
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://"):
        # Prefer explicit driver
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql+psycopg://"):
        # Map psycopg v3 DSN to psycopg2 since psycopg2-binary is installed
        return url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql+asyncpg://"):
        # Fallback to psycopg2 driver if asyncpg is not desired
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    return url


engine = create_engine(_normalize_database_url(settings.database_url), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

