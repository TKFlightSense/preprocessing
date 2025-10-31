from typing import List, Optional
from pydantic import BaseModel, Field


class LabelInstruction(BaseModel):
    labels: List[str] = Field(..., min_items=2, max_items=50)
    multi_label: bool = False
    require_confidence: bool = True


class ModelConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None


class LLMResponse(BaseModel):
    predicted_labels: List[str]
    confidences: Optional[List[float]] = None
    rationale: Optional[str] = None


class LabeledRow(BaseModel):
    idx: int
    text: str
    response: LLMResponse
    raw_json: dict
    prompt_hash: str
