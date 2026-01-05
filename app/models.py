from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


# ==============================
# USER & AUTH MODELS
# ==============================

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True


class UserCreate(UserBase):
    password: str = Field(..., min_length=6)


class UserInDB(UserBase):
    id: str
    hashed_password: str
    created_at: datetime
    last_login: Optional[datetime] = None


class UserResponse(UserBase):
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)


# ==============================
# SENSOR DATA MODELS
# ==============================

class SensorReading(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    mx: float
    my: float
    mz: float
    amb: float
    obj: float
    soc: Optional[int] = None
    millis: int = 0


class SensorDocument(BaseModel):
    time: datetime
    tag_id: int = Field(..., alias="tagID")
    sensor_data: List[SensorReading] = Field(..., alias="sensorData")
    rssi: Optional[float] = None
    snr: Optional[float] = None

    class Config:
        populate_by_name = True


# ==============================
# ANALYSIS MODELS
# ==============================

class DailyData(BaseModel):
    date: str
    activity_score: float
    activity_score_raw: Optional[float] = None
    resting_fraction: float
    eating_fraction: float
    ruminating_fraction: float
    walking_fraction: float
    estrus_confirmed: bool
    amb_mean: Optional[float] = None
    obj_mean: Optional[float] = None


class AnalysisSummary(BaseModel):
    avg_activity_score: float
    avg_activity_score_raw: Optional[float] = None
    avg_walking_fraction: float
    avg_resting_fraction: float
    max_activity_score: Optional[float] = None
    min_activity_score: Optional[float] = None
    max_activity_score_raw: Optional[float] = None
    min_activity_score_raw: Optional[float] = None


class AnalysisResponse(BaseModel):
    tag_id: str
    daily_data: List[DailyData]
    estrus_detected: bool
    estrus_date: Optional[str] = None
    total_days: int
    total_readings: int
    summary: AnalysisSummary
    last_updated: str


class CowStatus(BaseModel):
    tag_id: str
    last_reading_time: str
    battery_level: Optional[int] = None
    signal_strength: Optional[float] = None
    current_activity: Optional[str] = None
    estrus_status: bool = False


# ==============================
# SETTINGS MODELS
# ==============================

class SettingsUpdate(BaseModel):
    rolling_activity_days: Optional[int] = None
    estrus_baseline_days: Optional[int] = None
    relative_std_multiplier: Optional[float] = None
    absolute_activity_multiplier: Optional[float] = None
    walking_threshold: Optional[float] = None
    dominance_ratio_threshold: Optional[float] = None
    ws_refresh_interval: Optional[int] = None


class SettingsResponse(BaseModel):
    rolling_activity_days: int
    estrus_baseline_days: int
    relative_std_multiplier: float
    absolute_activity_multiplier: float
    walking_threshold: float
    dominance_ratio_threshold: float
    ws_refresh_interval: int


# ==============================
# WEBSOCKET MODELS
# ==============================

class WSMessage(BaseModel):
    type: str  # "update", "alert", "error", "connected"
    data: Any
    timestamp: str


class EstrusAlert(BaseModel):
    tag_id: str
    detected_date: str
    activity_score: float
    walking_fraction: float
    message: str
