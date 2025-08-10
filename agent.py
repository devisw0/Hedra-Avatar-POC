from dotenv import load_dotenv
import os, json, asyncio
import requests

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, RunContext, function_tool
from livekit.plugins.openai.realtime import RealtimeModel
from livekit.plugins import hedra

load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

def _fetch_current_weather(city: str, units: str = "imperial") -> dict:
    """
    Call OpenWeather "Current weather data" endpoint.
    Docs: https://openweathermap.org/current
    Example: http(s)://api.openweathermap.org/data/2.5/weather?q=Boston&units=imperial&appid=KEY
    """
    if not OPENWEATHER_KEY:
        raise RuntimeError("Missing OPENWEATHER_API_KEY in .env")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_KEY, "units": units}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()  # raise if 4xx/5xx
    return r.json()

class Assistant(Agent):
    # Expose the function as an LLM-callable tool
    @function_tool(
        description=(
            "Get current weather by city name. Use when the user asks about weather. "
            "Arguments: 'city' (e.g., 'Boston, US') and optional 'units' ('imperial' for °F, 'metric' for °C)."
        )
    )
    async def get_weather(
        self,
        context: RunContext,
        city: str,
        units: str = "imperial",
    ) -> dict:
        """
        Returns a concise JSON with city, temperature, conditions, humidity, wind, and unit system.
        """
        try:
            data = _fetch_current_weather(city, units)
            # Normalize a small, clean payload back to the model
            out = {
                "city": f"{data.get('name')}",
                "conditions": data["weather"][0]["description"] if data.get("weather") else "unknown",
                "temp": data["main"]["temp"] if data.get("main") else None,
                "feels_like": data["main"].get("feels_like") if data.get("main") else None,
                "humidity": data["main"].get("humidity") if data.get("main") else None,
                "wind_mph": data["wind"]["speed"] if data.get("wind") else None,
                "units": units,
            }
            return out
        except requests.HTTPError as e:
            # Return a clear tool error so the model can apologize or ask for another city
            return {"error": f"OpenWeather error: {e.response.status_code} {e.response.text[:120]}"}
        except Exception as e:
            return {"error": f"Weather lookup failed: {str(e)}"}

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a concise, helpful voice assistant. "
                "When the user asks about weather, call the 'get_weather' tool. "
                "If units are not specified, default to Fahrenheit (imperial). "
                "Be brief and mention temp and condition."
            )
            # Note: tools defined with @function_tool are auto-registered for this Agent.
        )

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
    await session.generate_reply(instructions="Greet the user briefly and tell them they can ask for weather in any city.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
