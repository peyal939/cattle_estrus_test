import argparse
import asyncio
from typing import Optional

from app.database import connect_to_mongo, close_mongo_connection, get_all_tag_ids, fetch_sensor_data, upsert_daily_metrics
from app.analysis import compute_daily_metrics


async def run(tag_id: Optional[str], days: int, start_date: Optional[str], end_date: Optional[str]):
    await connect_to_mongo()
    try:
        tag_ids = [tag_id] if tag_id else await get_all_tag_ids()
        if not tag_ids:
            print("No tags found")
            return

        # For now, precompute by reading raw data for the requested window.
        # This is intended to be run offline/periodically, not on every dashboard request.
        for tid in tag_ids:
            print(f"Precomputing daily metrics for tag {tid}...")

            # The raw fetch supports UTC datetime bounds; for this CLI we keep it simple and use days.
            # If you need date-bounded recompute, run this with --days 0 and restrict via DB queries.
            raw_df = await fetch_sensor_data(tag_id=str(tid), days=days)
            if raw_df.empty:
                print(f"  - No raw data")
                continue

            daily = compute_daily_metrics(raw_df)
            if daily is None or daily.empty:
                print(f"  - No daily rows computed")
                continue

            await upsert_daily_metrics(tag_id=str(tid), daily_rows=daily.to_dict(orient="records"))
            print(f"  - Upserted {len(daily)} days")

    finally:
        await close_mongo_connection()


def main():
    parser = argparse.ArgumentParser(description="Precompute daily_metrics for fast dashboard queries")
    parser.add_argument("--tag", dest="tag_id", default=None, help="Tag ID (omit for all tags)")
    parser.add_argument("--days", dest="days", type=int, default=60, help="How many days of raw data to aggregate (0 = all)")
    parser.add_argument("--start-date", dest="start_date", default=None, help="(reserved) YYYY-MM-DD BST")
    parser.add_argument("--end-date", dest="end_date", default=None, help="(reserved) YYYY-MM-DD BST")
    args = parser.parse_args()

    asyncio.run(run(args.tag_id, args.days, args.start_date, args.end_date))


if __name__ == "__main__":
    main()
