import logging

from chroma_db_manager import ChromaDBManager
from .models.voice_info import VoiceInfo

def calculate_cost(summary):
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

    logging.info(f"Usage: {summary}")
    logging.info(f"Costs: LLM=${llm_cost:.6f}, TTS=${tts_cost:.6f}, STT=${stt_cost:.6f}, Total=${total_cost:.6f}")

def get_language(lang: str) -> str:
    """Returns the language code based on the provided language string.
    """
    if lang.lower() == "it":
        return "Italiano"
    elif lang.lower() == "en":
        return "English"
    return "English"  # Default to English if no match found

def perform_rag(user_input: str, chroma_db_manager: ChromaDBManager) -> str:
    # Increased n_results to retrieve more potentially relevant documents
    user_input = user_input.replace(" ", "").replace("\n", "").replace("\t", "")
    retrieved_docs = chroma_db_manager.query_db(query_texts=user_input, n_results=5)
    context = "\n\n --- \n\n".join(doc[0].page_content for doc in retrieved_docs) if retrieved_docs else "No relevant information found."
    return context

def get_voice_info(lang: str) -> VoiceInfo:
    """Returns the language code based on the provided language string.
    """
    if lang.lower() == "it":
        return VoiceInfo(lang="it-IT", voice_name="it-IT-Standard-F", gender="male")
    elif lang.lower() == "en":
        return VoiceInfo(lang="en-US", voice_name="en-US-Standard-I", gender="male") 
    return VoiceInfo(lang="en-US", voice_name="en-US-Standard-I", gender="male") 