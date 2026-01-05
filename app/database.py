from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from app.config import get_settings, to_bst, format_bst
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
import pytz

settings = get_settings()


class Database:
    client: AsyncIOMotorClient = None


db = Database()


async def connect_to_mongo():
    """Connect to MongoDB"""
    db.client = AsyncIOMotorClient(settings.mongo_uri)
    # Test connection
    try:
        await db.client.admin.command('ping')
        # Ensure daily-metrics index exists (fast tag/date lookups, unique upserts)
        try:
            await get_daily_collection().create_index([("tagID", 1), ("date", 1)], unique=True)
        except Exception:
            # Index creation failure should not prevent service start
            pass
        print(f"✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise


async def close_mongo_connection():
    """Close MongoDB connection"""
    if db.client:
        db.client.close()
        print("🔌 MongoDB connection closed")


def get_database():
    """Get database instance"""
    return db.client[settings.mongo_db]


def get_collection(name: str = None):
    """Get collection instance"""
    collection_name = name or settings.mongo_collection
    return get_database()[collection_name]


def get_daily_collection():
    """Daily aggregates collection (one doc per tag per day)."""
    return get_database()[settings.mongo_daily_collection]


async def upsert_daily_metrics(tag_id: str, daily_rows: List[dict]) -> None:
    """Upsert daily metrics rows into MongoDB."""
    if not daily_rows:
        return
    col = get_daily_collection()
    from pymongo import UpdateOne

    tid = int(tag_id) if str(tag_id).isdigit() else tag_id
    ops = []
    for row in daily_rows:
        date_str = row.get("date")
        if not date_str:
            continue
        doc = dict(row)
        doc["tagID"] = tid
        doc["date"] = str(date_str)
        ops.append(
            UpdateOne({"tagID": tid, "date": str(date_str)}, {"$set": doc}, upsert=True)
        )
    if ops:
        await col.bulk_write(ops, ordered=False)


async def fetch_daily_metrics(
    tag_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: int = 30,
) -> pd.DataFrame:
    """Fetch daily metrics for a tag.

    Dates are YYYY-MM-DD strings in BST.
    If start/end are not provided and days>0, uses last N days by date string.
    If days==0 and no start/end, returns all available for the tag.
    """
    col = get_daily_collection()
    tid = int(tag_id) if str(tag_id).isdigit() else tag_id

    q: dict = {"tagID": tid}
    if start_date or end_date or (days and days > 0):
        # Compute implicit date range in BST when days provided.
        if not start_date and not end_date and days and days > 0:
            bst = pytz.timezone(settings.timezone)
            start_dt = (datetime.now(bst) - timedelta(days=days)).date()
            start_date = start_dt.isoformat()
        date_filter: dict = {}
        if start_date:
            date_filter["$gte"] = str(start_date)
        if end_date:
            date_filter["$lte"] = str(end_date)
        q["date"] = date_filter

    cursor = col.find(q).sort("date", 1)
    docs = await cursor.to_list(length=None)
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    return df


def get_users_collection():
    """Get users collection"""
    return get_database()["users"]


# ==============================
# USER DATABASE OPERATIONS
# ==============================

async def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username"""
    collection = get_users_collection()
    user = await collection.find_one({"username": username})
    if user:
        user["id"] = str(user["_id"])
    return user


async def get_user_by_id(user_id: str) -> Optional[dict]:
    """Get user by ID"""
    from bson import ObjectId
    collection = get_users_collection()
    user = await collection.find_one({"_id": ObjectId(user_id)})
    if user:
        user["id"] = str(user["_id"])
    return user


async def create_user(user_data: dict) -> dict:
    """Create a new user"""
    collection = get_users_collection()
    user_data["created_at"] = datetime.utcnow()
    user_data["last_login"] = None
    result = await collection.insert_one(user_data)
    user_data["id"] = str(result.inserted_id)
    return user_data


async def update_user_last_login(username: str):
    """Update user's last login time"""
    collection = get_users_collection()
    await collection.update_one(
        {"username": username},
        {"$set": {"last_login": datetime.utcnow()}}
    )


async def get_all_users() -> List[dict]:
    """Get all users (admin only)"""
    collection = get_users_collection()
    users = []
    async for user in collection.find():
        user["id"] = str(user["_id"])
        users.append(user)
    return users


async def delete_user(user_id: str) -> bool:
    """Delete a user by ID"""
    from bson import ObjectId
    collection = get_users_collection()
    result = await collection.delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count > 0


async def count_admins() -> int:
    """Count admin users"""
    collection = get_users_collection()
    return await collection.count_documents({"role": "admin"})


# ==============================
# SENSOR DATA OPERATIONS
# ==============================

async def fetch_sensor_data(
    tag_id: str = None,
    days: int = 30,
    start_utc: Optional[datetime] = None,
    end_utc: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch sensor data from MongoDB and flatten the nested sensorData array.
    All timestamps are converted to Bangladesh Standard Time (UTC+6).
    
    Note: The MongoDB collection has a quirk where the time field is stored as '"time"'
    (with literal quotes) instead of 'time'. We handle both cases.
    """
    collection = get_collection()

    # Base filter
    base_query: dict = {}
    if tag_id:
        base_query["tagID"] = int(tag_id) if str(tag_id).isdigit() else tag_id
    
    # Time filter (in UTC for MongoDB query)
    # Handle both '"time"' (with quotes) and 'time' field names.
    # Stored datetimes are typically naive UTC, so we query using naive UTC as well.
    # If caller doesn't provide an end bound, cap at "now" (+ a small skew)
    # to avoid future-dated garbage records (e.g., device clock issues like year 2036).
    if end_utc is None:
        end_utc = datetime.utcnow() + timedelta(minutes=5)

    # If no start_utc provided and days > 0, use days as the lookback window.
    # If days is 0, fetch all available data (no start filter), but still cap at now.
    if start_utc is None and days and days > 0:
        start_utc = datetime.utcnow() - timedelta(days=days)

    if start_utc is not None and start_utc.tzinfo is not None:
        start_utc = start_utc.astimezone(pytz.utc).replace(tzinfo=None)
    if end_utc is not None and end_utc.tzinfo is not None:
        end_utc = end_utc.astimezone(pytz.utc).replace(tzinfo=None)

    time_filter: dict = {}
    if start_utc is not None:
        time_filter["$gte"] = start_utc
    if end_utc is not None:
        time_filter["$lte"] = end_utc

    # Prefer querying the dominant time field without $or for better index usage.
    # Fallback to the alternate field name if no docs found.
    time_field_candidates = ['"time"', 'time']

    # Fetch only needed fields to reduce payload.
    projection = {
        '"time"': 1,
        'time': 1,
        'tagID': 1,
        'sensorData': 1,
        'rssi': 1,
        'snr': 1,
    }
    
    flattened_data = []
    bst = pytz.timezone(settings.timezone)

    used_time_field = None
    cursor = None
    for tf in time_field_candidates:
        query = dict(base_query)
        query[tf] = time_filter

        cursor = collection.find(query, projection=projection).sort(tf, 1)
        # If we can quickly detect emptiness, do it; otherwise stream and break once we see docs.
        first = await cursor.to_list(length=1)
        if first:
            used_time_field = tf
            # Re-create cursor for full iteration (the previous cursor is now advanced).
            cursor = collection.find(query, projection=projection).sort(tf, 1)
            break

    if used_time_field is None or cursor is None:
        return pd.DataFrame()

    async for doc in cursor:
        base_time = doc.get(used_time_field) or doc.get('"time"') or doc.get('time')
        if base_time is None:
            continue

        if isinstance(base_time, str):
            base_time = pd.to_datetime(base_time)

        if base_time.tzinfo is None:
            base_time = pytz.utc.localize(base_time)

        base_time_bst = base_time.astimezone(bst)

        tag = doc.get("tagID")
        rssi = doc.get("rssi")
        snr = doc.get("snr")

        sensor_readings = doc.get("sensorData", [])

        for reading in sensor_readings:
            millis_offset = reading.get("millis", 0)
            actual_time = base_time_bst + timedelta(milliseconds=millis_offset)

            flattened_data.append({
                "time": actual_time,
                "tag_id": tag,
                "ax": reading.get("ax", 0),
                "ay": reading.get("ay", 0),
                "az": reading.get("az", 0),
                "gx": reading.get("gx", 0),
                "gy": reading.get("gy", 0),
                "gz": reading.get("gz", 0),
                "mx": reading.get("mx", 0),
                "my": reading.get("my", 0),
                "mz": reading.get("mz", 0),
                "amb": reading.get("amb", 0),
                "obj": reading.get("obj", 0),
                "soc": reading.get("soc"),
                "rssi": rssi,
                "snr": snr,
            })
    
    df = pd.DataFrame(flattened_data)
    
    if df.empty:
        return df
    
    # Create date column
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date
    
    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)
    
    return df


async def get_all_tag_ids() -> List[str]:
    """Get list of all unique tag IDs"""
    collection = get_collection()
    tag_ids = await collection.distinct("tagID")
    return [str(tag_id) for tag_id in tag_ids]


async def get_latest_reading(tag_id: str = None) -> Optional[dict]:
    """Get the most recent sensor reading for a tag"""
    collection = get_collection()

    base_query: dict = {}
    if tag_id:
        base_query["tagID"] = int(tag_id) if str(tag_id).isdigit() else tag_id

    now_cap = datetime.utcnow() + timedelta(minutes=5)
    projection = {
        '"time"': 1,
        'time': 1,
        'tagID': 1,
        'sensorData': 1,
        'rssi': 1,
        'snr': 1,
    }

    # Try dominant field first for performance, fallback if none.
    for tf in ['"time"', 'time']:
        query = dict(base_query)
        query[tf] = {"$lte": now_cap}
        doc = await collection.find_one(query, sort=[(tf, DESCENDING)], projection=projection)
        if doc:
            break
    
    if not doc:
        return None
    
    # Convert time to BST - handle both field names
    time_utc = doc.get('"time"') or doc.get("time")
    if isinstance(time_utc, str):
        time_utc = pd.to_datetime(time_utc)
    
    time_bst = format_bst(time_utc) if time_utc else None
    
    sensor_data = doc.get("sensorData", [])
    last_reading = sensor_data[-1] if sensor_data else {}
    
    return {
        "time": time_bst,
        "tag_id": str(doc.get("tagID")),
        "readings_count": len(sensor_data),
        "rssi": doc.get("rssi"),
        "snr": doc.get("snr"),
        "battery_level": last_reading.get("soc"),
        "last_reading": last_reading
    }


async def get_reading_count(tag_id: str = None, days: int = 30) -> int:
    """Get count of readings for a tag"""
    collection = get_collection()
    
    query = {}
    if tag_id:
        query["tagID"] = int(tag_id) if tag_id.isdigit() else tag_id
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Handle both field name formats
    query["$or"] = [
        {'"time"': {"$gte": start_date}},
        {"time": {"$gte": start_date}}
    ]
    
    return await collection.count_documents(query)
