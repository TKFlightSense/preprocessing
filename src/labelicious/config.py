import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Env:
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("LABELICIOUS_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("LABELICIOUS_TEMPERATURE", 0.0))
    top_p: float = float(os.getenv("LABELICIOUS_TOP_P", 1.0))

ENV = Env()