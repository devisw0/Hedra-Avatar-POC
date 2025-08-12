from dotenv import load_dotenv
import os, json, asyncio, math
import requests
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
import country_converter as coco
from geopy.geocoders import Nominatim

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, RunContext, function_tool
from livekit.plugins.openai.realtime import RealtimeModel
from livekit.plugins import hedra

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
INTERNS_CSV_PATH = os.getenv("INTERNS_CSV_PATH", "./IFF Interns Information Form(Sheet1).csv")

# ---------------------------
# Helpers: geocoding + weather
# ---------------------------

def _geocode_city(q: str) -> dict | None:
    if not OPENWEATHER_KEY:
        raise RuntimeError("Missing OPENWEATHER_API_KEY in .env")
    q = q.strip()
    if not q:
        return None
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": q, "limit": 1, "appid": OPENWEATHER_KEY}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    arr = r.json() or []
    if not arr:
        return None
    item = arr[0]
    parts = [p for p in [item.get("name"), item.get("state"), item.get("country")] if p]
    return {
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "name": ", ".join(parts) if parts else q,
    }

def _fetch_current_weather_latlon(lat: float, lon: float, units: str = "imperial") -> dict:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def _fetch_forecast_latlon(lat: float, lon: float, units: str = "imperial") -> dict:
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------------------------
# Interns data layer
# ---------------------------

CONTINENTS = {
    "africa","antarctica","asia","europe","north america","south america","oceania"
}
SPORT_SYNONYMS = {"football": "soccer", "soccer/football": "soccer"}

def _norm(s):
    if s is None:
        return None
    if isinstance(s, float) and math.isnan(s):
        return None
    return str(s).strip()

def _canon(s):
    s = _norm(s)
    return s.lower() if s else None

def _title(s):
    s = _norm(s)
    return s.title() if s else None

def _split_multi(val):
    s = _canon(val)
    if not s: 
        return []
    parts = []
    for tok in s.replace("/", ",").replace(" and ", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        parts.append(t)
    return parts

def _canon_sport(s):
    if s is None: 
        return None
    s = s.strip().lower()
    return SPORT_SYNONYMS.get(s, s)

class InternDirectory:
    def __init__(self, csv_path: str):
        self.rows = []
        self.by_name = {}
        self.idx = {
            "sport": defaultdict(list),
            "team": defaultdict(list),
            "music": defaultdict(list),
            "hobby": defaultdict(list),
            "destination_region": defaultdict(list),
        }
        self._cc = coco.CountryConverter()
        self._geolocator = Nominatim(user_agent="interns_qna")  # online for city→country
        self._geocache = {}  # city->country
        self._load(csv_path)

    def _destination_to_continent(self, dest: str | None) -> str | None:
        if not dest: 
            return None
        d = dest.strip()
        if not d:
            return None
        dl = d.lower()
        if dl in CONTINENTS:
            return dl.title()
        # country → continent (offline)
        cont = self._cc.convert(d, to='continent')
        if cont and cont != 'not found':
            return cont.title()
        # try as city → country (online, cached)
        if d in self._geocache:
            country = self._geocache[d]
        else:
            try:
                loc = self._geolocator.geocode(d, exactly_one=True, timeout=5)
                country = None
                if loc and loc.address:
                    parts = [p.strip() for p in loc.address.split(",")]
                    country = parts[-1] if parts else None
                self._geocache[d] = country
            except Exception:
                country = None
                self._geocache[d] = None
        if country:
            cont2 = self._cc.convert(country, to='continent')
            if cont2 and cont2 != 'not found':
                return cont2.title()
        return None

    def _parse_sport_team(self, cell: str | None):
        sports = []
        teams = []
        for piece in _split_multi(cell):
            normalized = _canon_sport(piece)
            if any(k in piece for k in ["united","rcb","warriors"]):
                teams.append(piece)
            elif normalized:
                sports.append(normalized)
        return list(dict.fromkeys(sports)), list(dict.fromkeys(teams))

    def _load(self, csv_path: str):
        df = pd.read_csv(csv_path, header=None)
        headers = [ _norm(x) for x in df.iloc[0].tolist() ]
        data = df.iloc[1:].reset_index(drop=True)

        def find_col(name_contains):
            for i, h in enumerate(headers):
                if not h: 
                    continue
                if name_contains.lower() in h.lower():
                    return i
            return None

        col_id = find_col("Id") or 0
        col_name = find_col("Name") or 1
        col_music = find_col("Favorite Music") or 2
        col_show = find_col("Movie/Show") or 3
        col_sport = find_col("sport") or 4  # "Favorite sport  and/or favorite team"
        col_hobbies = find_col("Hobbies") or 5
        col_dest = find_col("travel destination") or 6
        col_food = find_col("Favorite Food") or 7

        for _, row in data.iterrows():
            rid = _norm(row[col_id])
            name = _norm(row[col_name])
            if name and name.strip().lower() == "devam":  # safety: prefer Devan
                name = "Devan"
            music = _norm(row[col_music])
            show = _norm(row[col_show])
            sport_cell = _norm(row[col_sport])
            hobbies_cell = _norm(row[col_hobbies])
            dest_cell = _norm(row[col_dest])
            food = _norm(row[col_food])

            sport_list, team_list = self._parse_sport_team(sport_cell)
            if sport_cell and "warriors" in sport_cell.lower() and "basketball" in sport_cell.lower():
                if "basketball" not in sport_list:
                    sport_list.append("basketball")
                if "warriors" not in team_list:
                    team_list.append("warriors")
            hobbies_list = _split_multi(hobbies_cell)
            destination_list = _split_multi(dest_cell) or ([dest_cell] if dest_cell else [])
            destination_regions = list({ self._destination_to_continent(d) for d in destination_list if d })
            destination_regions = [r for r in destination_regions if r]

            rec = {
                "id": rid,
                "full_name": name,
                "first_name": name.split()[0] if name else None,
                "last_name": name.split()[1] if (name and len(name.split())>1) else None,
                "music_genre": _canon(music),
                "music_display": _title(music),
                "tv_show": _norm(show),
                "sport_list": sport_list,
                "team_list": [t.lower() for t in team_list],
                "hobbies_list": hobbies_list,
                "destination_list": destination_list,
                "destination_regions": [r.lower() for r in destination_regions],
                "favorite_food": _norm(food),
            }
            self.rows.append(rec)
            self.by_name[name.lower()] = rec

        for rec in self.rows:
            nm = rec["full_name"]
            for s in rec["sport_list"]:
                self.idx["sport"][s].append(nm)
            for t in rec["team_list"]:
                self.idx["team"][t].append(nm)
            if rec["music_genre"]:
                self.idx["music"][rec["music_genre"]].append(nm)
            for h in rec["hobbies_list"]:
                self.idx["hobby"][h.lower()].append(nm)
            for r in rec["destination_regions"]:
                self.idx["destination_region"][r].append(nm)

    # utilities
    def fuzzy_find_name(self, q: str) -> str | None:
        names = [r["full_name"] for r in self.rows]
        match = rf_process.extractOne(q, names, scorer=rf_fuzz.WRatio, score_cutoff=80)
        return match[0] if match else None

    def list_by_sport(self, sport: str):
        return list(dict.fromkeys(self.idx["sport"].get(_canon_sport(sport), [])))

    def list_by_region_or_place(self, region_or_place: str):
        q = _canon(region_or_place)
        if not q:
            return []
        if q in CONTINENTS:
            return list(dict.fromkeys(
                sum([self.idx["destination_region"].get(q, [])], [])
            ))
        cont = self._destination_to_continent(region_or_place)
        if cont:
            return list(dict.fromkeys(
                sum([self.idx["destination_region"].get(cont.lower(), [])], [])
            ))
        return []

    def popularity(self, field: str):
        field = field.lower()
        counts = Counter()
        if field in ("music","music_genre"):
            for rec in self.rows:
                if rec["music_genre"]:
                    counts[rec["music_genre"]] += 1
        elif field in ("sport","sports"):
            for rec in self.rows:
                for s in rec["sport_list"]:
                    counts[s] += 1
        elif field in ("team","teams"):
            for rec in self.rows:
                for t in rec["team_list"]:
                    counts[t] += 1
        elif field in ("hobby","hobbies"):
            for rec in self.rows:
                for h in rec["hobbies_list"]:
                    counts[h.lower()] += 1
        elif field in ("destination","destination_region","region"):
            for rec in self.rows:
                for r in rec["destination_regions"]:
                    counts[r] += 1
        elif field in ("food","favorite_food"):
            for rec in self.rows:
                if rec["favorite_food"]:
                    counts[rec["favorite_food"].lower()] += 1
        elif field in ("show","tv","tv_show"):
            for rec in self.rows:
                if rec["tv_show"]:
                    counts[rec["tv_show"].lower()] += 1
        else:
            return {}
        total = sum(counts.values())
        return {"counts": counts.most_common(), "total": total}

# ---------------------------
# Agent
# ---------------------------

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a concise, helpful voice assistant.\n"
                "Weather tools: When the user asks about current weather, call 'get_weather'. "
                "When the user asks about future weather (e.g., tomorrow), call 'get_forecast'. "
                "- If a weather tool returns error 'missing_city', politely ask which city. "
                "- If the user omits a city but a previous city is known, reuse that city. "
                "If units are not specified, default to Fahrenheit (imperial). "
                "Be brief and mention temperature and condition.\n\n"
                "Interns tools: If the user asks 'who likes <sport>', call 'who_likes_sport'. "
                "If they ask 'who wants to travel to <place/region>' (e.g., Europe, Barcelona, Greece), call 'who_wants_to_travel'. "
                "If they ask 'what is <name>'s favorite <field>' (food, music, show, sport, team, hobby, destination), call 'get_favorite'. "
                "If they ask 'most popular <field>', call 'popularity'. "
                "If they ask for a short friendly description of a person, call 'creative_blurb'. "
                "Disambiguation: treat 'football' as 'soccer' unless the user specifies American football. "
                "If a name is ambiguous, ask for a quick clarification. "
                "Keep answers short and name the people clearly."
            )
        )
        # Session-scoped memory (weather)
        self.last_place = None
        self.last_units = "imperial"
        # Interns data
        self._interns = InternDirectory(INTERNS_CSV_PATH)

    # ---------- Intern tools ----------

    @function_tool(description="Return interns who like the given sport. Example: sport='soccer'")
    async def who_likes_sport(self, context: RunContext, sport: str) -> dict:
        people = self._interns.list_by_sport(sport or "")
        return {"sport": _canon_sport(sport), "people": people}

    @function_tool(description="Return interns who want to travel to a region or place (city/country/continent).")
    async def who_wants_to_travel(self, context: RunContext, region_or_place: str) -> dict:
        people = self._interns.list_by_region_or_place(region_or_place or "")
        return {"query": region_or_place, "people": people}

    @function_tool(description="Get a person's favorite field. field in {food, music, show, sport, team, hobby, destination}.")
    async def get_favorite(self, context: RunContext, name: str, field: str) -> dict:
        qname = _norm(name) or ""
        field = (field or "").strip().lower()
        rec = self._interns.by_name.get(qname.lower())
        if not rec:
            match = self._interns.fuzzy_find_name(qname)
            if match:
                rec = self._interns.by_name.get(match.lower())
                qname = match
        if not rec:
            return {"error": "not_found", "message": "I couldn't find that person."}
        if field in ("food","favorite_food"):
            val = rec["favorite_food"]
        elif field in ("music","music_genre"):
            val = rec["music_display"]
        elif field in ("show","tv","tv_show","movie"):
            val = rec["tv_show"]
        elif field in ("sport","sports"):
            val = ", ".join(rec["sport_list"]) if rec["sport_list"] else None
        elif field in ("team","teams"):
            val = ", ".join(rec["team_list"]) if rec["team_list"] else None
        elif field in ("hobby","hobbies"):
            val = ", ".join(rec["hobbies_list"]) if rec["hobbies_list"] else None
        elif field in ("destination","destination_region","region"):
            val = ", ".join(rec["destination_list"]) if rec["destination_list"] else None
        else:
            return {"error": "bad_field", "message": "Unknown field. Try food, music, show, sport, team, hobby, destination."}
        return {"name": rec["full_name"], "field": field, "value": val or "None listed"}

    @function_tool(description="Return popularity stats for a field (music, sport, team, hobby, destination_region, food, show).")
    async def popularity(self, context: RunContext, field: str) -> dict:
        stats = self._interns.popularity(field or "")
        if not stats:
            return {"error": "bad_field", "message": "Unknown field for popularity."}
        return stats

    @function_tool(description="Generic filter by interest: field + value (e.g., field='hobby', value='tennis').")
    async def find_by_interest(self, context: RunContext, field: str, value: str) -> dict:
        field = (field or "").strip().lower()
        val = (value or "").strip().lower()
        matches = []
        if field in ("music","music_genre"):
            matches = self._interns.idx["music"].get(val, [])
        elif field in ("sport","sports"):
            matches = self._interns.idx["sport"].get(_canon_sport(val), [])
        elif field in ("team","teams"):
            matches = self._interns.idx["team"].get(val, [])
        elif field in ("hobby","hobbies"):
            matches = self._interns.idx["hobby"].get(val, [])
        elif field in ("destination","destination_region","region"):
            matches = self._interns.list_by_region_or_place(value)
        else:
            return {"error": "bad_field", "message": "Unknown field. Try music, sport, team, hobby, destination."}
        return {"field": field, "value": value, "people": list(dict.fromkeys(matches))}

    @function_tool(description="Return a short 1–2 sentence friendly blurb about a person based on their data.")
    async def creative_blurb(self, context: RunContext, name: str) -> dict:
        rec = self._interns.by_name.get((name or "").strip().lower())
        if not rec:
            match = self._interns.fuzzy_find_name(name or "")
            if match:
                rec = self._interns.by_name.get(match.lower())
        if not rec:
            return {"error": "not_found", "message": "I couldn't find that person."}
        bits = []
        if rec["sport_list"]:
            s = ", ".join(rec["sport_list"])
            if rec["team_list"]:
                s += f" ({', '.join(rec['team_list'])})"
            bits.append(f"enjoys {s}")
        if rec["hobbies_list"]:
            bits.append(f"spends time on {', '.join(rec['hobbies_list'])}")
        if rec["favorite_food"]:
            bits.append(f"and never says no to {rec['favorite_food']}")
        if rec["destination_list"]:
            bits.append(f"Dream trip: {', '.join(rec['destination_list'])}.")
        text = f"{rec['full_name']} {'; '.join(bits)}" if bits else f"{rec['full_name']} has eclectic tastes."
        return {"name": rec["full_name"], "blurb": text}

    # ---------- Weather tools (existing) ----------

    @function_tool(description="Get current weather for a place; city like 'Hillsborough, NJ'. Units: imperial/metric.")
    async def get_weather(self, context: RunContext, city: str = "", units: str = "imperial") -> dict:
        units = (units or self.last_units or "imperial").lower()
        if units not in ("imperial", "metric"):
            units = "imperial"
        self.last_units = units
        place = None
        if (city or "").strip():
            place = _geocode_city(city)
            if not place:
                return {"error": "missing_city", "message": "I couldn’t find that place. Could you rephrase or add state/country?"}
            self.last_place = place
        elif self.last_place:
            place = self.last_place
        else:
            return {"error": "missing_city", "message": "Please tell me which city you'd like the weather for."}
        try:
            data = _fetch_current_weather_latlon(place["lat"], place["lon"], units)
            wind_speed = (data.get("wind") or {}).get("speed")
            wind_unit = "mph" if units == "imperial" else "m/s"
            out = {
                "resolved_place": place["name"],
                "conditions": (data.get("weather") or [{}])[0].get("description", "unknown"),
                "temp": (data.get("main") or {}).get("temp"),
                "feels_like": (data.get("main") or {}).get("feels_like"),
                "humidity": (data.get("main") or {}).get("humidity"),
                "wind_speed": wind_speed,
                "wind_unit": wind_unit,
                "units": units,
            }
            return out
        except requests.HTTPError as e:
            return {"error": f"OpenWeather error: {e.response.status_code} {e.response.text[:120]}"}
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}

    @function_tool(description="Get a simple forecast for 'tomorrow' for a place. Units: imperial/metric.")
    async def get_forecast(self, context: RunContext, city: str = "", units: str = "imperial", when: str = "tomorrow") -> dict:
        units = (units or self.last_units or "imperial").lower()
        if units not in ("imperial", "metric"):
            units = "imperial"
        self.last_units = units
        place = None
        if (city or "").strip():
            place = _geocode_city(city)
            if not place:
                return {"error": "missing_city", "message": "I couldn’t find that place. Could you rephrase or add state/country?"}
            self.last_place = place
        elif self.last_place:
            place = self.last_place
        else:
            return {"error": "missing_city", "message": "Please tell me which city you'd like the forecast for."}
        try:
            fc = _fetch_forecast_latlon(place["lat"], place["lon"], units)
            today_utc = datetime.utcnow().date()
            target_date = today_utc + timedelta(days=1)
            entries = []
            for item in fc.get("list", []):
                ts = item.get("dt_txt")
                if not ts:
                    continue
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                if dt.date() == target_date:
                    entries.append(item)
            if not entries:
                return {"error": "no_forecast", "message": "I couldn't find tomorrow’s forecast windows for that location."}
            temps = [ (e.get("main") or {}).get("temp") for e in entries if (e.get("main") or {}).get("temp") is not None ]
            temps = [t for t in temps if isinstance(t, (int, float))]
            descs = [ ((e.get("weather") or [{}])[0].get("description") or "").lower() for e in entries ]
            tmin = min(temps) if temps else None
            tmax = max(temps) if temps else None
            common_desc = (Counter([d for d in descs if d]).most_common(1)[0][0]) if any(descs) else "unknown"
            out = {
                "resolved_place": place["name"],
                "for_date": str(target_date),
                "summary": common_desc,
                "temp_min": tmin,
                "temp_max": tmax,
                "units": units,
            }
            return out
        except requests.HTTPError as e:
            return {"error": f"OpenWeather error: {e.response.status_code} {e.response.text[:120]}"}
        except Exception as e:
            return {"error": f"Forecast lookup failed: {str(e)}"}

# ---------------------------
# Worker entrypoint
# ---------------------------

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(llm=RealtimeModel())
    avatar = hedra.AvatarSession(avatar_id=os.getenv("HEDRA_AVATAR_ID"))
    await avatar.start(session, room=ctx.room)
    await session.start(room=ctx.room, agent=Assistant(), room_input_options=RoomInputOptions())
    await session.generate_reply(instructions="Hi there! You can ask me about weather or interns (e.g., 'who wants to travel to Europe?') or other questions.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
