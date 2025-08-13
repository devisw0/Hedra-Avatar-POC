#!/usr/bin/env python3
import os, asyncio, json
from pprint import pprint

# Import your agent from agent.py (must be in PYTHONPATH / same dir)
from agent import Assistant

async def main():
    asst = Assistant()

    print("\n=== Fragrance dataset readiness ===")
    ready = (not asst._frag_df.empty) and bool(asst._frag_cols)
    print("frag_ready:", ready)
    if not ready:
        print("HINT: set FRAGRANCE_CSV_PATH to your fra_cleaned.csv path")
        return

    # 1) Average rating over whole dataset
    print("\n--- frag_get_average(rating) ---")
    avg = await asst.frag_get_average(context=None, column="rating")
    pprint(avg)

    # 2) Top 5 by rating_count
    print("\n--- frag_get_top_n(rating_count, n=5) ---")
    top = await asst.frag_get_top_n(context=None, column="rating_count", n=5)
    pprint(top)

    # 3) Count by decade (if Year present)
    print("\n--- frag_count_items(decade) ---")
    counts = await asst.frag_count_items(context=None, group_by="decade")
    pprint(counts)

    # 4) Find by accords (rose/floral) if accords exist
    print("\n--- frag_find_by_notes(accords=['floral','woody']) ---")
    finds = await asst.frag_find_by_notes(context=None, notes=None, accords=['floral','woody'], match_all=False)
    pprint(finds)

    # 5) Optional weather tests (requires OPENWEATHER_API_KEY)
    if os.getenv("OPENWEATHER_API_KEY"):
        print("\n=== Weather checks ===")
        w = await asst.get_weather(context=None, city="Hillsborough, NJ", units="imperial")
        print("get_weather:", w)
        f = await asst.get_forecast(context=None, city="Hillsborough, NJ", units="imperial", when="tomorrow")
        print("get_forecast:", f)
    else:
        print("\n(Skipping weather tests; OPENWEATHER_API_KEY not set)")

if __name__ == "__main__":
    asyncio.run(main())
