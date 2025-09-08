"""
Clerk authentication middleware and utilities for FastAPI.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import requests
import jwt
from cryptography.hazmat.primitives import serialization
from .config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer()

JWKS_URL = f"{settings.clerk_issuer}/.well-known/jwks.json"
_jwks: Dict[str, Any] | None = None


def load_jwks() -> Dict[str, Any]:
    global _jwks
    if _jwks is None:
        try:
            resp = requests.get(JWKS_URL, timeout=5)
            resp.raise_for_status()
            _jwks = resp.json()
        except Exception as e:
            logger.error(f"Could not load JWKS: {e}")
            raise HTTPException(status_code=503, detail="Auth key fetch failed")
    return _jwks


def jwk_to_pem(jwk_key: Dict[str, Any]) -> str:
    # RSA n/e -> PEM
    import base64
    n_b64 = jwk_key.get("n")
    e_b64 = jwk_key.get("e")
    if not n_b64 or not e_b64:
        raise HTTPException(status_code=401, detail="Invalid JWK")

    def b64url_to_int(b64: str) -> int:
        pad = "=" * (-len(b64) % 4)
        return int.from_bytes(base64.urlsafe_b64decode(b64 + pad), "big")

    n_int = b64url_to_int(n_b64)
    e_int = b64url_to_int(e_b64)

    from cryptography.hazmat.primitives.asymmetric import rsa
    pub = rsa.RSAPublicNumbers(e_int, n_int).public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def get_public_key_pem(token: str) -> str:
    jwks = load_jwks()
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwk_to_pem(key)
    raise HTTPException(status_code=401, detail="Public key not found")


def verify_token(token: str) -> Dict[str, Any]:
    try:
        public_key_pem = get_public_key_pem(token)
        payload = jwt.decode(
            token,
            public_key_pem,
            algorithms=["RS256"],
            issuer=settings.clerk_issuer,
            options={"verify_aud": False},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidIssuerError:
        raise HTTPException(status_code=401, detail="Invalid issuer")
    except jwt.InvalidTokenError as e:
        logger.error(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


class ClerkUser:
    """Represents an authenticated Clerk user."""
    
    def __init__(self, user_id: str, email: Optional[str] = None, 
                 first_name: Optional[str] = None, last_name: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.metadata = metadata or {}
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        elif self.last_name:
            return self.last_name
        else:
            return "Unknown User"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.full_name,
            "metadata": self.metadata
        }


def extract_user_from_payload(payload: Dict[str, Any]) -> ClerkUser:
    """Extract user information from JWT payload."""
    try:
        user_id = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing user ID"
            )
        # Prefer direct claims typical in Clerk JWTs
        email = payload.get("email_address") or payload.get("email")
        if not email:
            # Fallback to possible arrays
            emails = payload.get("email_addresses") or []
            if isinstance(emails, list) and emails:
                email = emails[0] if isinstance(emails[0], str) else emails[0].get("email_address")
        first_name = payload.get("first_name") or payload.get("given_name")
        last_name = payload.get("last_name") or payload.get("family_name")
        metadata = {k: v for k, v in payload.items() if k not in {"sub", "iss", "aud", "exp", "iat", "nbf"}}
        return ClerkUser(
            user_id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"Failed to extract user from payload: {str(e)} | payload={payload}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user data in token"
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> ClerkUser:
    """
    Dependency to get the current authenticated user.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        ClerkUser: The authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        print('credentials', credentials)
        if not credentials:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated")
        token = credentials.credentials
        if not token:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authenticated")
        payload = verify_token(token)
        print('payload', payload)
        # Extract user
        user = extract_user_from_payload(payload)
        print('user', user)
        logger.info(f"User authenticated: {user.user_id}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[ClerkUser]:
    """
    Dependency to get the current user if authenticated, None otherwise.
    Useful for endpoints that work with or without authentication.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_auth():
    """Decorator to require authentication for endpoints."""
    def decorator(func):
        func.__requires_auth__ = True
        return func
    return decorator


# Utility functions for user session management
def generate_user_session_id(user_id: str, session_id: str) -> str:
    """Generate a user-scoped session ID."""
    return f"{user_id}:{session_id}"


def parse_user_session_id(user_session_id: str) -> tuple[str, str]:
    """Parse user session ID to extract user_id and session_id."""
    try:
        user_id, session_id = user_session_id.split(":", 1)
        return user_id, session_id
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user session ID format"
        )


def validate_user_session_access(user: ClerkUser, session_id: str) -> bool:
    """Validate that a user can access a specific session."""
    # For now, we'll implement basic validation
    # In a more complex system, you might check database permissions
    try:
        user_id, _ = parse_user_session_id(session_id)
        return user_id == user.user_id
    except HTTPException:
        return False
