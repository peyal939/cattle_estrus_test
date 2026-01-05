"""
Check exact structure of harnesstag.iotdata
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta

MONGO_URI = "mongodb+srv://r-fine:m0ng0ap$10t@apsiot.xq0fdv2.mongodb.net"

async def check_data():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["harnesstag"]
    collection = db["iotdata"]
    
    # Get sample document
    print("📄 Sample document structure:")
    sample = await collection.find_one()
    
    for key, value in sample.items():
        print(f"  {repr(key)}: {type(value).__name__} = {repr(value)[:100]}")
    
    # Get unique tagIDs
    print("\n🏷️ Unique tagIDs:")
    tag_ids = await collection.distinct("tagID")
    print(f"  {tag_ids}")
    
    # Get date range
    print("\n📅 Date range:")
    
    # Try different time field names
    for time_field in ['time', '"time"', 'timestamp']:
        try:
            oldest = await collection.find_one(sort=[(time_field, 1)])
            newest = await collection.find_one(sort=[(time_field, -1)])
            if oldest and time_field in oldest:
                print(f"  Field '{time_field}':")
                print(f"    Oldest: {oldest.get(time_field)}")
                print(f"    Newest: {newest.get(time_field)}")
        except:
            pass
    
    # Count recent documents
    print("\n📊 Document counts:")
    total = await collection.count_documents({})
    print(f"  Total: {total}")
    
    # Check for documents with 'time' field
    with_time = await collection.count_documents({"time": {"$exists": True}})
    print(f"  With 'time' field: {with_time}")
    
    with_quoted_time = await collection.count_documents({'"time"': {"$exists": True}})
    print(f"  With '\"time\"' field: {with_quoted_time}")
    
    client.close()

if __name__ == "__main__":
    asyncio.run(check_data())
