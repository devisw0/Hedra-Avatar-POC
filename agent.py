from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins.openai.realtime import RealtimeModel
from livekit.plugins import hedra

load_dotenv()

import sys
for k in ["OPENAI_API_KEY","LIVEKIT_URL","LIVEKIT_API_KEY","LIVEKIT_API_SECRET","HEDRA_API_KEY","HEDRA_AVATAR_ID"]:
    if not os.getenv(k):
        sys.exit(f"Missing required env var: {k}")
        
assert os.getenv("LIVEKIT_URL"), "LIVEKIT_URL missing"
assert os.getenv("HEDRA_AVATAR_ID"), "HEDRA_AVATAR_ID missing"
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY missing"

class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a concise, helpful voice assistant.")

async def entrypoint(ctx: agents.JobContext):
    # OpenAI Realtime = STT + LLM + TTS
    session = AgentSession(llm=RealtimeModel())  # uses OPENAI_API_KEY

    # Hedra avatar video (lip-syncs to the agent's audio)
    avatar = hedra.AvatarSession(avatar_id=os.getenv("HEDRA_AVATAR_ID"))
    await avatar.start(session, room=ctx.room)

    # Start the agent
    await session.start(room=ctx.room, agent=Assistant(), room_input_options=RoomInputOptions())

    # Optional: greet once
    await session.generate_reply(instructions="Greet the user briefly.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
