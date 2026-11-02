"""Thin wrapper around a local LM Studio server (Requirement 4's LLM reasoning layer).

LM Studio exposes an OpenAI-compatible chat-completions endpoint, so this uses
the ``openai`` SDK pointed at it rather than a bespoke HTTP client.

Configuration is via two environment variables (set these before running
anything in ``healthy.llm``, no defaults are safe to assume):

    LLM_BASE_URL  e.g. http://127.0.0.1:1234/v1
        LM Studio picks a fresh port on each launch unless pinned in its
        settings -- copy the exact URL shown in LM Studio's "Developer" /
        server tab when you start the server.
    LLM_MODEL     the exact model identifier shown in that same tab
        (e.g. "qwen2.5-7b-instruct"), not a display name.
"""

from __future__ import annotations

import json
import os

from pydantic import BaseModel

_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def _client():
    from openai import OpenAI

    base_url = os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL)
    return OpenAI(base_url=base_url, api_key="lm-studio")


def _model_name() -> str:
    model = os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError(
            "LLM_MODEL environment variable is not set -- set it to the exact "
            "model identifier shown in LM Studio's server tab."
        )
    return model


def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
) -> dict:
    """Call the local LLM and return its response parsed against ``schema``.

    Uses LM Studio's structured-output support (``response_format`` with a
    JSON schema) so the model is constrained to the shape of ``schema`` at
    generation time. If the first response still isn't valid JSON -- local
    models occasionally wrap output in prose or markdown fences despite the
    schema constraint -- retries once with an explicit "JSON only" follow-up
    before giving up.
    """
    client = _client()
    model = _model_name()

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
        },
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
    ).choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": "That was not valid JSON. Respond with ONLY the JSON object, no prose or markdown fences.",
            }
        )
        retry_content = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
        ).choices[0].message.content
        return json.loads(retry_content)
