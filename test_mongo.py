"""
Quick test script to check MongoDB data
"""
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta

MONGO_URI = "mongodb+srv://r-fine:m0ng0ap$10t@apsiot.xq0fdv2.mongodb.net"

async def test_connection():
    print("Connecting to MongoDB...")
    client = AsyncIOMotorClient(MONGO_URI)
    
    # List all databases
    print("\n📂 Available databases:")
    dbs = await client.list_database_names()
    for db in dbs:
        print(f"  - {db}")
    
    # Check each database for collections
    for db_name in dbs:
        if db_name not in ['admin', 'local', 'config']:
            db = client[db_name]
            collections = await db.list_collection_names()
            print(f"\n📁 Collections in '{db_name}':")
            for coll in collections:
                count = await db[coll].count_documents({})
                print(f"  - {coll} ({count} documents)")
                
                # Show sample document
                if count > 0:
                    sample = await db[coll].find_one()
                    print(f"    Sample keys: {list(sample.keys())}")
                    if 'time' in sample:
                        print(f"    Time field: {sample['time']}")
                    if 'tagID' in sample:
                        print(f"    tagID: {sample['tagID']}")
    
    client.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(test_connection())
