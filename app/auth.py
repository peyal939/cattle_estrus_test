from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt

from app.config import get_settings
from app.models import (
    UserCreate, UserResponse, Token, TokenData,
    LoginRequest, UserRole, PasswordChange
)
from app.user_db import (
    get_user_by_username, create_user, update_user_last_login,
    get_all_users, delete_user, count_admins, get_user_by_id
)

settings = get_settings()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# Router
router = APIRouter(prefix="/auth", tags=["Authentication"])


# ==============================
# PASSWORD UTILITIES
# ==============================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return bcrypt.checkpw(
        plain_password.encode('utf-8'), 
        hashed_password.encode('utf-8')
    )


def get_password_hash(password: str) -> str:
    """Hash a password"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


# ==============================
# JWT UTILITIES
# ==============================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None:
            return None
        
        return TokenData(username=username, role=role)
    except JWTError:
        return None


# ==============================
# AUTHENTICATION DEPENDENCIES
# ==============================

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Get the current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
    
    token_data = decode_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = get_user_by_username(token_data.username)
    if user is None:
        raise credentials_exception
    
    if not user.get("is_active", True):
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Get current active user"""
    return current_user


async def get_admin_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Require admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


async def get_optional_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """Get user if authenticated, else None"""
    if not token:
        return None
    
    token_data = decode_token(token)
    if token_data is None:
        return None
    
    return get_user_by_username(token_data.username)


# ==============================
# AUTH ROUTES
# ==============================

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    update_user_last_login(user["username"])
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=str(user["id"]),
            username=user["username"],
            email=user.get("email"),
            full_name=user.get("full_name"),
            role=user["role"],
            is_active=bool(user.get("is_active", True)),
            created_at=datetime.fromisoformat(user["created_at"]) if isinstance(user["created_at"], str) else user["created_at"],
            last_login=datetime.fromisoformat(user["last_login"]) if user.get("last_login") else None
        )
    )


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest):
    """Login with username and password"""
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    
    update_user_last_login(user["username"])
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=UserResponse(
            id=str(user["id"]),
            username=user["username"],
            email=user.get("email"),
            full_name=user.get("full_name"),
            role=user["role"],
            is_active=bool(user.get("is_active", True)),
            created_at=datetime.fromisoformat(user["created_at"]) if isinstance(user["created_at"], str) else user["created_at"],
            last_login=datetime.fromisoformat(user["last_login"]) if user.get("last_login") else None
        )
    )


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    current_user: dict = Depends(get_admin_user)
):
    """Register a new user (admin only)"""
    # Check if username exists
    existing = get_user_by_username(user_data.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create user
    user_dict = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "role": user_data.role.value if hasattr(user_data.role, 'value') else user_data.role,
        "is_active": user_data.is_active,
        "hashed_password": get_password_hash(user_data.password)
    }
    
    new_user = create_user(user_dict)
    
    return UserResponse(
        id=str(new_user["id"]),
        username=new_user["username"],
        email=new_user.get("email"),
        full_name=new_user.get("full_name"),
        role=new_user["role"],
        is_active=bool(new_user.get("is_active", True)),
        created_at=new_user["created_at"],
        last_login=None
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """Get current user information"""
    return UserResponse(
        id=str(current_user["id"]),
        username=current_user["username"],
        email=current_user.get("email"),
        full_name=current_user.get("full_name"),
        role=current_user["role"],
        is_active=bool(current_user.get("is_active", True)),
        created_at=datetime.fromisoformat(current_user["created_at"]) if isinstance(current_user["created_at"], str) else current_user["created_at"],
        last_login=datetime.fromisoformat(current_user["last_login"]) if current_user.get("last_login") else None
    )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: dict = Depends(get_current_active_user)
):
    """Change current user's password"""
    if not verify_password(password_data.current_password, current_user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    from app.user_db import get_db
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET hashed_password = ? WHERE username = ?",
            (get_password_hash(password_data.new_password), current_user["username"])
        )
    
    return {"message": "Password changed successfully"}


# ==============================
# ADMIN ROUTES
# ==============================

@router.get("/users", response_model=list[UserResponse])
async def list_users(current_user: dict = Depends(get_admin_user)):
    """List all users (admin only)"""
    users = get_all_users()
    return [
        UserResponse(
            id=str(user["id"]),
            username=user["username"],
            email=user.get("email"),
            full_name=user.get("full_name"),
            role=user["role"],
            is_active=bool(user.get("is_active", True)),
            created_at=datetime.fromisoformat(user["created_at"]) if isinstance(user["created_at"], str) else user["created_at"],
            last_login=datetime.fromisoformat(user["last_login"]) if user.get("last_login") else None
        )
        for user in users
    ]


@router.delete("/users/{user_id}")
async def remove_user(user_id: str, current_user: dict = Depends(get_admin_user)):
    """Delete a user (admin only)"""
    user_id_int = int(user_id)
    
    # Prevent self-deletion
    if user_id_int == current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    # Check if this would delete the last admin
    user_to_delete = get_user_by_id(user_id_int)
    
    if user_to_delete and user_to_delete.get("role") == "admin":
        admin_count = count_admins()
        if admin_count <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the last admin account"
            )
    
    success = delete_user(user_id_int)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}


# ==============================
# HELPER FUNCTIONS
# ==============================

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate a user by username and password"""
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
