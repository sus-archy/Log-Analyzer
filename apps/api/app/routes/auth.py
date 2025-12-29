"""
Authentication routes.

Provides token-based authentication endpoints.
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from ..core.config import settings
from ..core.logging import get_logger
from ..core.security import (
    authenticate_user,
    create_access_token,
    get_current_user,
    User,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["authentication"])


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRequest(BaseModel):
    """Token request for JSON body auth."""
    username: str
    password: str


class UserResponse(BaseModel):
    """User info response."""
    username: str
    auth_enabled: bool


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2 compatible token login.
    
    Returns a JWT access token for authenticated users.
    
    - **username**: Admin username (default: admin)
    - **password**: Admin password
    """
    if not settings.auth_enabled:
        # If auth is disabled, return a token for any user
        access_token = create_access_token(
            data={"sub": form_data.username or "anonymous"},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        return Token(
            access_token=access_token,
            expires_in=settings.access_token_expire_minutes * 60
        )
    
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    
    logger.info(f"User {user.username} logged in successfully")
    
    return Token(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/token/json", response_model=Token)
async def login_json(request: TokenRequest):
    """
    JSON body token login (alternative to form-based).
    
    For clients that prefer JSON over form data.
    """
    if not settings.auth_enabled:
        access_token = create_access_token(
            data={"sub": request.username or "anonymous"},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        return Token(
            access_token=access_token,
            expires_in=settings.access_token_expire_minutes * 60
        )
    
    user = authenticate_user(request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    
    return Token(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get current authenticated user info.
    
    Returns username and auth status.
    """
    return UserResponse(
        username=current_user.username if current_user else "anonymous",
        auth_enabled=settings.auth_enabled
    )


@router.get("/status")
async def auth_status():
    """
    Check authentication status.
    
    Returns whether auth is enabled and token expiration settings.
    """
    return {
        "auth_enabled": settings.auth_enabled,
        "token_expire_minutes": settings.access_token_expire_minutes,
        "rate_limit_enabled": settings.rate_limit_enabled,
    }
