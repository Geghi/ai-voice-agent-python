import logging
import os
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import json

logger = logging.getLogger("agent")

load_dotenv(".env.local")

SYSTEM_PROMPT = """
You are a friendly and skilled AI language tutor. Your purpose is to help the user improve their spoken ${language} fluency through natural, engaging, and context-aware conversation.

**Your Role:**  
- Act as a supportive **conversation partner**, not a strict teacher.  
- Speak naturally, as if in a real-time voice chat.

**Conversation Strategy:**  
1. **Focus on Fluency:**  
   - Encourage the user to express ideas freely.  
   - Avoid over-correcting; prioritize confidence and flow.

2. **Natural Tone:**  
   - Use friendly, concise, and casual language.  
   - Keep responses short and open-ended to stimulate more speaking.

3. **Personalization:**  
   - The user is interested in: **${interests}**.  
   - Weave these interests naturally into conversation.  
   - Ask **open-ended questions** related to these topics.

4. **Subtle Corrections:**  
   - Gently rephrase any grammatical mistakes in your replies **without pointing them out**.  
   - Example: If the user says "She go school", respond with: "Oh nice, she goes to school every day?"

5. **Vocabulary Building:**  
   - Introduce new, useful words naturally.  
   - Be ready to explain them simply if the user asks.

6. **Conversation Flow:**  
   - Keep the dialogue moving.  
   - If a topic ends, shift smoothly to a **related or interest-based** topic.

ask open-ended questions to keep the user engaged and speaking.

EXTREMELY IMPORTANT: Do not EVER, for ANY Reason use a different language than ${language}, even if the user is speaking another language, always respond in ${language}.
    """

class Assistant(Agent):
    def __init__(self, interests: str = "") -> None:
        base_instructions = SYSTEM_PROMPT.format(
            interests=interests, 
            language="English"
        )

        super().__init__(
            instructions=base_instructions,
        )

    # all functions annotated with @function_tool will be passed to the LLM when this
    # agent is active
    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """

        logger.info(f"Looking up weather for {location}")

        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    logger.info("Entrypoint function started.")
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    
    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel()
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()


    async def log_usage():
        summary = usage_collector.get_summary()
        
        # Define pricing for each service
        # Google Text-to-Speech (Standard Voice): $4.00 per 1 million characters
        TTS_PRICE_PER_MILLION_CHARS = 4.00
        # Deepgram Nova-3 (STT): $0.0045 per minute
        STT_PRICE_PER_MINUTE = 0.0045
        # OpenAI GPT-4o-mini (LLM)
        LLM_INPUT_PRICE_PER_MILLION_TOKENS = 0.15
        LLM_CACHED_PRICE_PER_MILLION_TOKENS = 0.075
        LLM_OUTPUT_PRICE_PER_MILLION_TOKENS = 0.60

        # Calculate costs
        llm_cost = (summary.llm_prompt_tokens / 1_000_000) * LLM_INPUT_PRICE_PER_MILLION_TOKENS + \
                   (summary.llm_completion_tokens / 1_000_000) * LLM_OUTPUT_PRICE_PER_MILLION_TOKENS + \
                   (summary.llm_prompt_cached_tokens / 1_000_000) * LLM_CACHED_PRICE_PER_MILLION_TOKENS
        
        tts_cost = (summary.tts_characters_count / 1_000_000) * TTS_PRICE_PER_MILLION_CHARS
        
        stt_cost = (summary.stt_audio_duration / 60) * STT_PRICE_PER_MINUTE

        total_cost = llm_cost + tts_cost + stt_cost

        logger.info(f"Usage: {summary}")
        logger.info(f"Costs: LLM=${llm_cost:.6f}, TTS=${tts_cost:.6f}, STT=${stt_cost:.6f}, Total=${total_cost:.6f}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://livekit.io/docs/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://livekit.io/docs/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    
    def _on_participant_connected(participant):
        logger.info(f"[_on_participant_connected] Function called for participant: {participant.identity}")
        if participant.metadata:
            logging.info(f"Participant metadata found: {participant.metadata}")
            try:
                metadata_json = json.loads(participant.metadata)
                interests = metadata_json.get("interests", "")
                if interests:
                    return interests
                else:
                    logging.info("No 'interests' key found in metadata for this participant.")
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode participant metadata as JSON for {participant.identity}.")
        else:
            logging.info(f"No metadata available for participant: {participant.identity}")

    logger.info("Registering participant_connected event listener.")
    # Register the event listener before connecting to ensure we catch all participants
    ctx.room.on("participant_connected", _on_participant_connected)

    # Join the room and connect to the user
    logger.info("Connecting to room.")
    await ctx.connect()
    logger.info("Connected to room.")

    # Check for participants already in the room when the agent connects
    logger.info("Checking for existing participants in the room.")
    
    interests = ""
    # Process local participant if it has metadata
    local_participant = ctx.room.local_participant
    if local_participant and local_participant.metadata:
        logger.info(f"Processing local participant: {local_participant.identity}")
        interests = _on_participant_connected(local_participant)

    # Process remote participants
    for participant in ctx.room.remote_participants.values():
        logger.info(f"Processing existing remote participant: {participant.identity}")
        interests = _on_participant_connected(participant)
    logger.info("Finished checking for existing participants.")

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        # tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # Google Cloud Text-to-Speech v1 Standard voices do not support streaming synthesis.
        # Setting use_streaming to False to allow the use of standard voices.
        tts=google.TTS(language="en-US", gender="female", voice_name="en-US-Standard-H", credentials_info=json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")), use_streaming=False),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://livekit.io/docs/agents/how-tos/turn-detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        user_away_timeout=20.0, # Disconnect after 20 seconds of no input
    )


    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(interests=interests),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
