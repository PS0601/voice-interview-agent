import os
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.agents.voice import room_io
from livekit.plugins import (
    langchain,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from graph import create_workflow

load_dotenv(Path(__file__).parent / ".env.local")

# Optional: set AGENT_NAME in .env.local to switch to explicit dispatch
# (the agent then only joins rooms it is dispatched to). Empty = joins every new room,
# which is the easiest way to test from the LiveKit playground.
AGENT_NAME = os.getenv("AGENT_NAME", "")

# Optional: set BEY_AVATAR_ID (and BEY_API_KEY) to enable the Beyond Presence avatar.
BEY_AVATAR_ID = os.getenv("BEY_AVATAR_ID", "")


class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a professional interviewer conducting a job interview. "
                "The LangGraph workflow will drive the conversation flow. "
                "Be conversational, professional, and helpful throughout the interview process."
            )
        )


server = AgentServer()


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):

    # Create the LangGraph workflow
    lg_llm = langchain.LLMAdapter(graph=create_workflow())

    # Configure the realtime voice pipeline
    session = AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
        ),
        llm=lg_llm,
        tts=cartesia.TTS(
            model=os.getenv("CARTESIA_MODEL", "sonic-3"),
            voice=os.getenv("CARTESIA_VOICE_ID", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # Optional Beyond Presence avatar (voice-only if BEY_AVATAR_ID is not set)
    if BEY_AVATAR_ID:
        from livekit.plugins import bey

        avatar = bey.AvatarSession(avatar_id=BEY_AVATAR_ID)
        await avatar.start(session, room=ctx.room)

    # Start the agent session
    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    print("Interview agent started.")


if __name__ == "__main__":
    cli.run_app(server)
