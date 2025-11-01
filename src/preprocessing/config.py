import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Env:
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model: str = os.getenv("PREPROCESSING_MODEL", "gpt-5-mini")
    temperature: float = float(os.getenv("PREPROCESSING_TEMPERATURE", 1.0))
    top_p: float = float(os.getenv("PREPROCESSING_TOP_P", 1.0))


ENV = Env()
