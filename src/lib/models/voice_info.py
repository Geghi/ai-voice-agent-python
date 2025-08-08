from pydantic import BaseModel

class VoiceInfo(BaseModel):
    lang: str
    gender: str
    voice_name: str