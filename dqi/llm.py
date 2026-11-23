"""Tiny client for a local Ollama model (free, runs on your laptop)."""

from __future__ import annotations

import os

import requests

DEFAULT_MODEL = os.getenv("DQI_MODEL", "qwen3:4b")
DEFAULT_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Room for the prompt + several tool results. Lower it (e.g. 4096) if your Mac runs out of memory.
CONTEXT_TOKENS = int(os.getenv("DQI_CONTEXT", "6144"))
# Qwen3 "thinking" makes tool choices much more reliable on small models but is slower.
# Turn it off with DQI_THINK=0 if runs take too long.
THINK = os.getenv("DQI_THINK", "1").lower() not in ("0", "false", "no")


class LLMError(Exception):
    """The model could not be reached or returned something unusable."""


class OllamaClient:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 timeout: int = 300):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> dict:
        """Send the conversation; return the assistant message dict.

        json_mode=True asks Ollama to force a valid JSON reply (used when the
        model's final answer was not valid JSON).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": CONTEXT_TOKENS},  # temperature 0 = repeatable
        }
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["format"] = "json"
        if self.model.startswith("qwen3"):
            payload["think"] = THINK

        try:
            response = requests.post(f"{self.url}/api/chat", json=payload,
                                     timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.url}. Start the Ollama app "
                "(or run `ollama serve`) and try again."
            ) from exc
        except requests.Timeout as exc:
            raise LLMError(f"Ollama did not answer within {self.timeout}s.") from exc

        if response.status_code == 404:
            raise LLMError(
                f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )
        if not response.ok:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")

        message = response.json().get("message")
        if not isinstance(message, dict):
            raise LLMError("Ollama returned a response without a message.")
        return message
