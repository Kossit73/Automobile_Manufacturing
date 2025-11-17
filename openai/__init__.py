"""Lightweight OpenAI SDK shim for offline environments.

This module mirrors the minimal interface used by the app:
`from openai import OpenAI` and subsequent `OpenAI(...).chat.completions.create(...)`.
It uses the OpenAI HTTP API directly so deployments without the official SDK
can still authenticate and run chat completions when an API key is supplied.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from urllib import request

DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


class _ChatCompletions:
    def __init__(self, client: "OpenAI"):
        self._client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> SimpleNamespace:
        if not self._client.api_key:
            raise RuntimeError("OpenAI API key is missing; set OPENAI_API_KEY or pass api_key explicitly")

        payload = json.dumps({"model": model, "messages": messages, "temperature": temperature}).encode(
            "utf-8"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._client.api_key}",
        }
        req = request.Request(self._client.base_url.rstrip("/") + "/chat/completions", data=payload, headers=headers)
        try:
            with request.urlopen(req, timeout=30) as resp:  # nosec B310 - controlled URL
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network specific
            raise RuntimeError(f"OpenAI REST request failed: {exc}")

        try:
            content = body["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - response validation
            raise RuntimeError(f"Unexpected OpenAI response: {exc}")

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _Chat:
    def __init__(self, client: "OpenAI"):
        self.completions = _ChatCompletions(client)


class OpenAI:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **_: Any):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or DEFAULT_BASE_URL
        self.chat = _Chat(self)

__all__ = ["OpenAI"]
