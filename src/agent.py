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
from chroma_db_manager import ChromaDBManager
from lib.utils import calculate_cost, get_language, get_voice_info, perform_rag
from lib.prompts import SYSTEM_PROMPT
from livekit.agents.voice.generation import update_instructions, INSTRUCTIONS_MESSAGE_ID

logger = logging.getLogger("agent")
load_dotenv(".env.local")



class Assistant(Agent):
    def __init__(self, language: str = "EN") -> None:
        self.language = get_language(language)
        base_instructions = SYSTEM_PROMPT.format(
            rag_context="\r\n --- \r\n\n",
            language=self.language,
        )

        super().__init__(
            instructions=base_instructions,
        )
        self.chroma_db_manager = ChromaDBManager(path="./chroma/curriculum-mantovani_chroma_db")

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        logger.info("on_user_turn_completed called.")
        logger.info(f"\r\n\r\nNew message content: {new_message}\r\n\r\n")
        logger.info(f"\r\r\r\ncurrent chat context: {turn_ctx}\r\n\r\n")
        user_input = ""
        if isinstance(new_message.content, str):
            user_input = new_message.content
        elif isinstance(new_message.content, list):
            for item in new_message.content:
                if isinstance(item, str):
                    user_input = item
                    break
                elif hasattr(item, 'text'):
                    user_input = item.text
                    break
                    
        if user_input:
            logger.info(f"\r\n\r\nUser input received: {user_input}\r\n\r\n")
            rag_context = perform_rag(user_input, self.chroma_db_manager)
            
            augmented_instructions = SYSTEM_PROMPT.format(
                rag_context=rag_context,
                language=self.language
            )
            
            update_instructions(turn_ctx, instructions=augmented_instructions, add_if_missing=True)

    @function_tool
    async def get_my_skills(self, context: RunContext):
        """Use this tool to answer any questions about my technical background, capabilities, or experience with specific technologies, programming languages, frameworks, databases, cloud providers, or infrastructure services (e.g., 'Have you used -blank-?', 'What are your skills?', 'Do you know -blank-?'). This tool should be called for any inquiry related to my technical proficiencies, regardless of the exact phrasing."""
        logging.info(f"\r\n\r\nRetrieving Giacomo Mantovani's skills.\r\n\r\n")

        return """My technical skills and Tools I Use (Strategically, Not Just Buzzwords) includes:
            llms / ai: OpenAI, Claude, Gemini, LangChain, LangGraph, RAG, Embeddings, Text to Speech, Speech to Text, OCR, AI Vision
            vector dbs: Pinecone, ChromaDB
            backend: FastAPI, Azure Functions, AWS Lambda
            frontend: Next.js, React, SvelteKit, Tailwind CSS
            database and cloud: Supabase, PostgreSQL, Cloudflare, Vercel, MongoDB
        """
    
    @function_tool
    async def recent_achievements(self, context: RunContext):
        """Use this tool to get a list of my recent achievements."""
        logging.info(f"\r\n\r\nRetrieving Giacomo Mantovani's recent achievements.\r\n\r\n")
        return """My recent achievements include:
            80%+ manual time reduction via AI auto-validated pipeline (no human in the loop)
            AI System Classifying from 400+ unique ticket types using LLM (Higher accuracy than manual classification)
            Co-founded AiTradigPredictor, building an LLM-powered financial prediction platform
        """
    
    @function_tool
    async def get_my_projects(self, context: RunContext):
        """Use this tool to get a list of my notable projects.

        This includes project names and a brief description of each.
        """
        projects = [
            "AITradingPredictor: AI platform for stock price predictions.",
            "Curriculum Voice Chatbot: Voice-based resume chatbot (You are using this one).",
            "AiLinkMind: Chatbot with web data scraping and integration.",
            "MiiroAI: AI assistant for meeting summaries and actions.",
            "Mavena: Salon CMS SaaS for bookings and analytics.",
            "AIGiftWhisper: Gift suggestions from chat.",
            "AI-Powered Inheritance Automation: Automates succession analysis.",
            "AI-Powered Bank Balances Automation: Extracts and analyzes bank data.",
            "High-Volume Ticketing System Automation: Automates enterprise ticketing."
        ]

        return "My notable projects include: " + "\n".join(projects) + "."


    @function_tool
    async def get_my_bio(self, context: RunContext):
        """Use this tool to get a brief biography of Giacomo Mantovani."""
        logging.info(f"\r\n\r\nRetrieving Giacomo Mantovani's biography.\r\n\r\n")
        return """I'm a Full Stack Engineer specialized in AI Solutions with over three years of experience in developing advanced Artificial Intelligence (AI) technologies. 
            My expertise includes working with Generative AI, AI Chatbots or Agents, RAG pipelines, and various AI technologies such as text-to-speech, speech-to-text, and AI vision.
            I have experience working with different AI frameworks and libraries, including LangChain, LangGraph, and OpenAI."""
        
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
        calculate_cost(summary)
        
    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://livekit.io/docs/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://livekit.io/docs/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    
    def _on_participant_connected(participant):
        if participant.metadata:
            try:
                metadata_json = json.loads(participant.metadata)
                language = metadata_json.get("language", "EN")
                if language:
                    return language
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode participant metadata as JSON for {participant.identity}.")

        return "EN"
    
    ctx.room.on("participant_connected", _on_participant_connected)

    # Join the room and connect to the user
    await ctx.connect()

    language = "EN"
    # Process local participant if it has metadata
    local_participant = ctx.room.local_participant
    if local_participant and local_participant.metadata:
        language = _on_participant_connected(local_participant)

    for participant in ctx.room.remote_participants.values():
        language = _on_participant_connected(participant)

    voice_data = get_voice_info(language)

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
        tts=google.TTS(language=voice_data.lang, gender=voice_data.gender, voice_name=voice_data.voice_name, credentials_info=json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")), use_streaming=False),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://livekit.io/docs/agents/how-tos/turn-detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        user_away_timeout=10.0, # Disconnect after 10 seconds of no input
    )


    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(language=language),
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
