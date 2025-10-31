from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import hashlib
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ENV


@dataclass
class LLMResult:
    content: str
    usage: dict


def prompt_hash(system: str, user: str, model: str) -> str:
    m = hashlib.sha256()
    m.update(system.encode("utf-8"))
    m.update(user.encode("utf-8"))
    m.update(model.encode("utf-8"))
    return m.hexdigest()[:16]


class OpenAIClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or ENV.openai_key
        self.base_url = base_url or ENV.openai_base
        self.model = model or ENV.model
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5)
    )
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> LLMResult:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            body["seed"] = seed
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResult(content=content, usage=usage)
