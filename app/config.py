from pydantic_settings import BaseSettings
from functools import lru_cache
from datetime import datetime
import pytz


class Settings(BaseSettings):
    # MongoDB
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "cattle_monitoring"
    mongo_collection: str = "sensor_data"
    mongo_daily_collection: str = "daily_metrics"
    
    # JWT Authentication
    jwt_secret: str = "change-this-secret-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    
    # Timezone (Bangladesh Standard Time UTC+6)
    timezone: str = "Asia/Dhaka"
    
    # WebSocket refresh interval (seconds)
    ws_refresh_interval: int = 30
    
    # Analysis Parameters
    rolling_activity_days: int = 3
    estrus_baseline_days: int = 7
    relative_std_multiplier: float = 1.5
    absolute_activity_multiplier: float = 1.25
    walking_threshold: float = 0.30
    dominance_ratio_threshold: float = 1.15
    
    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_bst_timezone():
    """Get Bangladesh Standard Time timezone object"""
    return pytz.timezone(get_settings().timezone)


def to_bst(dt: datetime) -> datetime:
    """Convert a datetime to Bangladesh Standard Time"""
    bst = get_bst_timezone()
    
    if dt is None:
        return None
    
    # If naive datetime, assume UTC
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    
    return dt.astimezone(bst)


def now_bst() -> datetime:
    """Get current time in Bangladesh Standard Time"""
    return datetime.now(get_bst_timezone())


def format_bst(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format a datetime in BST"""
    bst_dt = to_bst(dt)
    return bst_dt.strftime(fmt) if bst_dt else None
