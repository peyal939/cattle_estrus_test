from __future__ import annotations

from pymongo import MongoClient, DESCENDING

from app.config import get_settings


def main(tag_id: int = 1004) -> None:
    s = get_settings()
    client = MongoClient(s.mongo_uri)
    try:
        col = client[s.mongo_db][s.mongo_collection]

        quoted_time_key = '"time"'

        doc = col.find_one({"tagID": tag_id}, sort=[(quoted_time_key, DESCENDING)])
        if not doc:
            doc = col.find_one({"tagID": tag_id}, sort=[("time", DESCENDING)])

        if not doc:
            print(f"No document found for tag {tag_id}")
            return

        print("_id", doc.get("_id"))
        print("doc_keys", sorted(doc.keys()))

        base = doc.get(quoted_time_key) or doc.get("time")
        print("tagID", doc.get("tagID"))
        print("base_time", base, "type", type(base))
        print("has_quoted_time", quoted_time_key in doc, "has_time", "time" in doc)

        # Print any alternate time-like fields if present
        for candidate in ["timestamp", "ts", "createdAt", "created_at", "serverTime", "deviceTime"]:
            if candidate in doc:
                print(f"{candidate}", doc.get(candidate), "type", type(doc.get(candidate)))

        sd = doc.get("sensorData") or []
        ms = [r.get("millis") for r in sd if r.get("millis") is not None]

        print("sensorData_len", len(sd))
        if ms:
            print("millis_min", min(ms))
            print("millis_max", max(ms))
            print("millis_first10", ms[:10])

    finally:
        client.close()


if __name__ == "__main__":
    main()
