import asyncio, json
from agent import Assistant

async def main():
    a = Assistant()

    results = {
      "top_accords_france": await a.frag_top_components(None, kind="accords",
                            filters={"country": "France"}, top_n=20),
      "top_notes_france_women": await a.frag_top_components(None, kind="notes",
                            filters={"country": "France", "gender": "female"}, top_n=20),
      "top_accords_france_2010s": await a.frag_top_components(None, kind="accords",
                            filters={"country": "France", "decade": 2010}, top_n=15),
      "top_notes_fr_since2018_votes500": await a.frag_top_components(None, kind="notes",
                            filters={"country": "France", "year_min": 2018, "votes_min": 500}, top_n=15),
    }

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
