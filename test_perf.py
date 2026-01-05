import asyncio
import time
from app.database import connect_to_mongo, fetch_sensor_data, close_mongo_connection
from app.analysis import analyze_estrus

async def test():
    await connect_to_mongo()
    
    t0 = time.time()
    print('Fetching data...')
    df = await fetch_sensor_data('1001', days=30)
    print(f'Fetched {len(df)} rows in {time.time()-t0:.2f}s')
    
    t1 = time.time()
    print('Running analysis...')
    result = analyze_estrus(df)
    print(f'Analysis done in {time.time()-t1:.2f}s')
    print(f'Total days: {result["total_days"]}')
    
    await close_mongo_connection()

if __name__ == "__main__":
    asyncio.run(test())
