# agent_smoketest_fragrance.py
import asyncio, json
from agent import Assistant

async def main():
    a = Assistant()

    # Show what the CSV loader mapped (useful for debugging)
    print("=== Column map ===")
    print(a._frag_cols)
    print()

    results = {}

    # 1) Basic stats
    results["average_rating"] = await a.frag_get_average(None, column="rating")

    # 2) Top lists
    results["top5_by_rating"] = await a.frag_get_top_n(None, column="rating", n=5)
    results["top5_by_rating_votes1000"] = await a.frag_get_top_n(
        None, column="rating", n=5, filters={"votes_min": 1000}
    )
    results["top5_women_by_rating"] = await a.frag_get_top_n(
        None, column="rating", n=5, filters={"gender": "female"}
    )
    results["top5_chanel_by_rating_count"] = await a.frag_get_top_n(
        None, column="rating_count", n=5, filters={"brand": "Chanel"}
    )

    # 3) Notes / accords lookups
    results["find_notes_vanilla_or_tobacco"] = await a.frag_find_by_notes(
        None, notes=["vanilla", "tobacco"], filters={}, match_all=False  # any-match
    )
    results["find_accords_floral_and_woody_allmatch"] = await a.frag_find_by_notes(
        None, accords=["floral", "woody"], filters={}, match_all=True  # require both
    )

    # 4) Grouped counts
    results["counts_by_decade"] = await a.frag_count_items(None, group_by="decade")
    results["counts_by_country_unisex"] = await a.frag_count_items(
        None, group_by="country", filters={"gender": "unisex"}
    )

    # 5) Brand-scoped averages (compare houses)
    results["avg_rating_dior"] = await a.frag_get_average(
        None, column="rating", filters={"brand": "Dior"}
    )
    results["avg_rating_guerlain"] = await a.frag_get_average(
        None, column="rating", filters={"brand": "Guerlain"}
    )

    # 6) Price-based queries (only if your CSV actually has a price column mapped)
    if "price" in (a._frag_cols or {}):
        results["cheapest_mens_rating4_vanilla"] = await a.frag_get_cheapest(
            None, filters={"gender": "male", "rating_min": 4, "notes": ["vanilla"]}, match_all_notes=False
        )
        results["most_expensive_chanel"] = await a.frag_get_most_expensive(
            None, filters={"brand": "Chanel"}
        )
    else:
        results["cheapest_mens_rating4_vanilla"] = {"skipped": "no 'price' column in dataset"}
        results["most_expensive_chanel"] = {"skipped": "no 'price' column in dataset"}

    print("\n=== Fragrance smoketest results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
