"""
Security module - JWT authentication and authorization.

Provides:
- Password hashing with bcrypt
- JWT token generation and validation
- OAuth2 password bearer scheme
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme - token URL is relative to API root
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    username: str
    exp: datetime


class User(BaseModel):
    """User model for authentication."""
    username: str
    disabled: bool = False


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Payload data (should include 'sub' for username)
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=settings.algorithm
    )
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm]
        )
        username: str = payload.get("sub")
        exp = payload.get("exp")
        
        if username is None:
            return None
        
        return TokenData(username=username, exp=datetime.fromtimestamp(exp, tz=timezone.utc))
    
    except JWTError as e:
        logger.debug(f"Token decode failed: {e}")
        return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user with username and password.
    
    For now, uses simple environment-based credentials.
    In production, this should query a user database.
    
    Args:
        username: Username to authenticate
        password: Plain text password
        
    Returns:
        User if authenticated, None otherwise
    """
    # Simple auth: check against environment variables
    if username == settings.admin_username:
        if verify_password(password, settings.admin_password_hash):
            return User(username=username)
        # Also allow plain text comparison for initial setup
        if password == settings.admin_password_hash:
            logger.warning("Using plain text password comparison - please hash your password")
            return User(username=username)
    
    return None


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """
    Get current user from JWT token.
    
    This is a dependency that can be used in route handlers.
    Returns None if auth is disabled or no valid token.
    
    Args:
        token: JWT token from Authorization header
        
    Returns:
        User if authenticated, None if auth disabled
        
    Raises:
        HTTPException: If auth enabled but token invalid
    """
    # If auth is disabled, allow all requests
    if not settings.auth_enabled:
        return User(username="anonymous")
    
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = decode_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check expiration
    if token_data.exp < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return User(username=token_data.username)


async def require_auth(user: User = Depends(get_current_user)) -> User:
    """
    Require authentication - raises if not authenticated.
    
    Use this as a dependency for protected routes.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# Optional auth - returns user if authenticated, None otherwise
async def optional_auth(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[User]:
    """
    Optional authentication - doesn't raise if not authenticated.
    
    Use this for routes that work with or without auth.
    """
    if not settings.auth_enabled:
        return User(username="anonymous")
    
    if token is None:
        return None
    
    token_data = decode_token(token)
    if token_data is None:
        return None
    
    return User(username=token_data.username)
