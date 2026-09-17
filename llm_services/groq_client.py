"""Thin Groq API wrapper for the product-facing language layer.

Used for word/phrase suggestions and (later) TTS/STT — never inside the
measured decoder path in /decoder. See MASTER_PROJECT_DOCUMENT.md Section 13.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

CONFIG_ENV_PATH = Path(__file__).resolve().parent.parent / "config" / ".env"
load_dotenv(dotenv_path=CONFIG_ENV_PATH)

DEFAULT_SUGGESTION_MODEL = os.getenv("GROQ_WORD_SUGGESTION_MODEL", "openai/gpt-oss-20b")
DEFAULT_EXPANSION_MODEL = os.getenv("GROQ_PHRASE_EXPANSION_MODEL", "openai/gpt-oss-120b")


class GroqClient:
    """Wraps the Groq SDK with the project's default models and config source."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy config/.env.example to config/.env "
                "and fill in your key."
            )
        self._client = Groq(api_key=key)

    def chat_completion(
        self,
        prompt: str,
        model: str = DEFAULT_SUGGESTION_MODEL,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
