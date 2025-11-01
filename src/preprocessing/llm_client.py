from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import hashlib
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

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


def _is_transient(e: Exception) -> bool:
    
# For errors 429 and 5xx

    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        return code == 429 or 500 <= code < 600
    if isinstance(e, httpx.TimeoutException):
        return True
    return False


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
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_transient),
    )
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> LLMResult:
        base = self.base_url.rstrip("/")
        url = f"{base}/chat/completions"
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
            if resp.status_code >= 400:
                # Try to extract API error payload to aid debugging
                try:
                    detail = resp.json()
                except Exception:
                    detail = {"raw": resp.text[:2000]}

                msg = json.dumps(detail)
                # If the model rejects certain params (e.g., temperature/top_p/seed),
                # retry once without that param on the same endpoint.
                if resp.status_code == 400:
                    err = (detail or {}).get("error", {})
                    param = err.get("param")
                    code = err.get("code")
                    if code == "unsupported_value" and param in {"temperature", "top_p", "seed"}:
                        trimmed = dict(body)
                        trimmed.pop(param, None)
                        alt_same = client.post(url, headers=headers, json=trimmed)
                        if alt_same.status_code < 400:
                            data = alt_same.json()
                            content = data["choices"][0]["message"]["content"]
                            usage = data.get("usage", {})
                            return LLMResult(content=content, usage=usage)
                        # If the trimmed retry also fails, re-extract error for accurate reporting
                        try:
                            detail = alt_same.json()
                        except Exception:
                            detail = {"raw": alt_same.text[:2000]}
                        msg = json.dumps(detail)

                # Optional fallback: if the server indicates Chat Completions is not supported,
                # try the Responses API with a simple string input.
                if resp.status_code == 400 and (
                    "not supported" in msg.lower() or "use the responses api" in msg.lower()
                ):
                    alt_url = f"{base}/responses"
                    alt_body: Dict[str, Any] = {
                        "model": self.model,
                        "input": f"System:\n{system}\n\nUser:\n{user}",
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    if seed is not None:
                        alt_body["seed"] = seed

                    alt = client.post(alt_url, headers=headers, json=alt_body)
                    if alt.status_code >= 400:
                        try:
                            d2 = alt.json()
                        except Exception:
                            d2 = {"raw": alt.text[:2000]}
                        raise httpx.HTTPStatusError(
                            f"OpenAI error {alt.status_code} on /responses: {json.dumps(d2)}",
                            request=alt.request,
                            response=alt,
                        )
                    data = alt.json()
                    # Responses API: best-effort extraction of text and usage
                    content = (
                        (data.get("output") or {}).get("text")
                        or (data.get("choices") or [{}])[0].get("message", {}).get("content")
                        or alt.text
                    )
                    usage = data.get("usage", {})
                    return LLMResult(content=content, usage=usage)

                # No fallback taken: raise with details so caller sees the real error
                raise httpx.HTTPStatusError(
                    f"OpenAI error {resp.status_code} on /chat/completions: {msg}",
                    request=resp.request,
                    response=resp,
                )

            data = resp.json()

        # Chat Completions happy path
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResult(content=content, usage=usage)
