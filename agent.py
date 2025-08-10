from dotenv import load_dotenv
import os, json, asyncio
import requests
from datetime import datetime, timedelta
from collections import Counter

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, RunContext, function_tool
from livekit.plugins.openai.realtime import RealtimeModel
from livekit.plugins import hedra

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

# ---------------------------
# Helpers: geocoding + weather
# ---------------------------

def _geocode_city(q: str) -> dict | None:
    """
    Use OpenWeather Direct Geocoding to resolve a free-text place into lat/lon.
    Docs: https://openweathermap.org/api/geocoding-api
    Returns: {"lat": float, "lon": float, "name": "Hillsborough, NJ, US"} or None
    """
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
    # Build a friendly display name
    parts = [p for p in [item.get("name"), item.get("state"), item.get("country")] if p]
    return {
        "lat": item.get("lat"),
        "lon": item.get("lon"),
        "name": ", ".join(parts) if parts else q,
    }

def _fetch_current_weather_latlon(lat: float, lon: float, units: str = "imperial") -> dict:
    """
    Call Current Weather with precise coordinates.
    Docs: https://openweathermap.org/current
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def _fetch_forecast_latlon(lat: float, lon: float, units: str = "imperial") -> dict:
    """
    5 day / 3 hour forecast.
    Docs: https://openweathermap.org/forecast5
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------------------------
# Agent
# ---------------------------

class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a concise, helpful voice assistant. "
                "When the user asks about current weather, call 'get_weather'. "
                "When the user asks about future weather (e.g., tomorrow, next days), call 'get_forecast'. "
                "- If a weather tool returns error 'missing_city', politely ask which city. "
                "- If the user omits a city but a previous city is known, reuse that city. "
                "If units are not specified, default to Fahrenheit (imperial). "
                "Be brief and mention temperature and condition."
            )
            # Tools defined with @function_tool are auto-registered.
        )
        # Session-scoped memory
        self.last_place = None  # {"name": str, "lat": float, "lon": float}
        self.last_units = "imperial"

    # Expose: CURRENT weather by place (with geocoding)
    @function_tool(
        description=(
            "Get current weather for a place. Accepts free-form city/town like 'Hillsborough, NJ' "
            "and optional units ('imperial' or 'metric')."
        )
    )
    async def get_weather(
        self,
        context: RunContext,
        city: str = "",
        units: str = "imperial",
    ) -> dict:
        """
        Returns: city display name, temp, condition, humidity, wind, and units.
        If city missing, uses last known place if available; otherwise asks for city.
        """
        # Normalize units and remember preference
        units = (units or self.last_units or "imperial").lower()
        if units not in ("imperial", "metric"):
            units = "imperial"
        self.last_units = units

        # Resolve place
        place = None
        if city.strip():
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

    # Expose: simple "tomorrow" forecast summary
    @function_tool(
        description=(
            "Get a simple forecast for 'tomorrow' for a place. Accepts free-form city/town text "
            "and optional units ('imperial' or 'metric'). Summarizes min/max temp and common conditions."
        )
    )
    async def get_forecast(
        self,
        context: RunContext,
        city: str = "",
        units: str = "imperial",
        when: str = "tomorrow",  # kept for future extensibility
    ) -> dict:
        """
        Returns a concise summary for tomorrow: min/max temp and most common condition.
        Reuses last known place if city omitted.
        """
        # Normalize units and remember preference
        units = (units or self.last_units or "imperial").lower()
        if units not in ("imperial", "metric"):
            units = "imperial"
        self.last_units = units

        # Resolve place
        place = None
        if city.strip():
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

            # Compute "tomorrow" in UTC (simple approximation good enough for demo)
            today_utc = datetime.utcnow().date()
            target_date = today_utc + timedelta(days=1)

            # Filter 3-hour entries for the target date
            entries = []
            for item in fc.get("list", []):
                # item["dt_txt"] like "2025-08-10 12:00:00" (UTC)
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

            # Aggregate min/max temps and most common description
            temps = [ (e.get("main") or {}).get("temp") for e in entries if (e.get("main") or {}).get("temp") is not None ]
            descs = [ ((e.get("weather") or [{}])[0].get("description") or "").lower() for e in entries ]
            temps = [t for t in temps if isinstance(t, (int, float))]

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
    # OpenAI Realtime = STT + LLM + TTS
    session = AgentSession(
        llm=RealtimeModel(),  # uses OPENAI_API_KEY
    )

    # Hedra avatar video (lip-syncs to the agent's audio)
    avatar = hedra.AvatarSession(avatar_id=os.getenv("HEDRA_AVATAR_ID"))
    await avatar.start(session, room=ctx.room)

    # Start the agent
    await session.start(room=ctx.room, agent=Assistant(), room_input_options=RoomInputOptions())

    # Optional: greet once
    await session.generate_reply(instructions="Hi there! Ask me the current weather or the forecast for tomorrow in any city.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
