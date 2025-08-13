import os
import json
import asyncio
from collections import OrderedDict

# Make sure we import the Assistant from your agent module
from agent import Assistant

def _sort_counts(d: dict, *, key_numeric: bool = False, reverse: bool = True) -> dict:
    if not isinstance(d, dict):
        return d
    if key_numeric:
        # sort by numeric keys (e.g., decade)
        items = sorted(((int(k), v) for k, v in d.items() if str(k).isdigit()), key=lambda x: x[0], reverse=reverse)
        # include any non-numeric keys at the end, stable
        others = [(k, v) for k, v in d.items() if not str(k).isdigit()]
        items += others
        return OrderedDict((str(k), v) for k, v in items)
    else:
        # sort by value desc
        return OrderedDict(sorted(d.items(), key=lambda kv: kv[1], reverse=reverse))

async def run():
    a = Assistant()

    # Show what columns were detected/mapped
    print("=== Column map ===")
    print(a._frag_cols)
    print()

    results = {}

    # 1) Average rating (whole dataset)
    results["average_rating"] = await a.frag_get_average(None, column="rating")

    # 2) Top-5 by rating (raw)
    results["top5_by_rating"] = await a.frag_get_top_n(None, column="rating", n=5)

    # 3) Top-5 by rating with votes >= 1000 (quality threshold)
    results["top5_by_rating_votes1000"] = await a.frag_get_top_n(None, column="rating", n=5, filters={"votes_min": 1000})

    # 4) Top-5 women by rating
    results["top5_women_by_rating"] = await a.frag_get_top_n(None, column="rating", n=5, filters={"gender": "female"})

    # 5) Top-5 France + men by rating_count
    results["top5_france_men_by_rating_count"] = await a.frag_get_top_n(None, column="rating_count", n=5, filters={"country": "France", "gender": "male"})

    # 6) Find perfumes with vanilla OR tobacco notes
    results["find_notes_vanilla_or_tobacco"] = await a.frag_find_by_notes(None, notes=["vanilla", "tobacco"], match_all=False)

    # 7) Find perfumes with floral AND woody accords (all-match)
    results["find_accords_floral_and_woody_allmatch"] = await a.frag_find_by_notes(None, accords=["floral", "woody"], match_all=True)

    # 8) Counts by decade
    counts_decade = await a.frag_count_items(None, group_by="decade")
    counts_decade["counts"] = _sort_counts(counts_decade.get("counts", {}), key_numeric=True, reverse=True)
    results["counts_by_decade"] = counts_decade

    # 9) Counts by country for unisex
    counts_country_uni = await a.frag_count_items(None, group_by="country", filters={"gender": "unisex"})
    counts_country_uni["counts"] = _sort_counts(counts_country_uni.get("counts", {}), key_numeric=False, reverse=True)
    results["counts_by_country_unisex"] = counts_country_uni

    # 10) Average rating: Dior + Guerlain
    results["avg_rating_dior"] = await a.frag_get_average(None, column="rating", filters={"brand": "Dior"})
    results["avg_rating_guerlain"] = await a.frag_get_average(None, column="rating", filters={"brand": "Guerlain"})

    # 11) 1990s with citrus notes (sample filter)
    results["nineties_with_citrus"] = await a.frag_find_by_notes(None, notes=["citrus"], filters={"year_min": 1990, "year_max": 1999})

    # 12) French + women + jasmine notes
    results["french_women_jasmine"] = await a.frag_find_by_notes(None, notes=["jasmine"], filters={"country": "France", "gender": "female"})

    # 13) Bottom-10 by rating among items with 'oud' in notes
    results["bottom10_with_oud"] = await a.frag_get_bottom_n(None, column="rating", n=10, filters={"notes": ["oud"]})

    # 14) 2010s, woody accords, rating >= 4.0
    results["woody_2010s_rating_ge4"] = await a.frag_find_by_notes(None, accords=["woody"], filters={"decade": 2010, "rating_min": 4.0})

    # 15) Top-5 Jean-Paul Gaultier by rating_count
    results["top5_jpg_by_votes"] = await a.frag_get_top_n(None, column="rating_count", n=5, filters={"brand": "jean-paul-gaultier"})

    # Write results to a JSON file for easy diffing
    out_path = "fragrance_smoketest_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Fragrance smoketest results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved JSON to {out_path}")

if __name__ == "__main__":
    asyncio.run(run())
