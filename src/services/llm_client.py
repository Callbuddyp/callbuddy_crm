from __future__ import annotations

import copy
import functools
import os
import time
from typing import Iterable, Mapping, Optional, Protocol, Sequence

from openai import OpenAI
from utils import load_env_value


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect HTTP 429 rate-limit errors from any LLM provider."""
    if getattr(exc, "status_code", None) == 429:
        return True
    message = str(exc).lower()
    return "429" in message and ("rate limit" in message or "too many requests" in message)


def _retry_on_rate_limit(func):
    """Decorator: retry with exponential backoff on rate-limit (429) errors."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_attempts = 5
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if _is_rate_limit_error(exc) and attempt < max_attempts:
                    print(f"[llm] Rate limited (attempt {attempt}/{max_attempts}); retrying in {delay}s …")
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
    return wrapper


def _sanitize_schema_for_fireworks(schema: dict) -> dict:
    """Transform a JSON Schema into a Fireworks-compatible subset.

    Fireworks structured output does not support:
      - ``oneOf`` (use ``anyOf`` instead)
      - ``"type": ["string", "null"]`` array syntax
      - ``null`` inside ``enum`` arrays

    This function recursively walks the schema and rewrites these constructs.
    The original schema is not mutated; a deep copy is returned.
    """
    schema = copy.deepcopy(schema)

    def _walk(node: dict) -> dict:
        if not isinstance(node, dict):
            return node

        # 1. oneOf → anyOf
        if "oneOf" in node:
            node["anyOf"] = node.pop("oneOf")

        # 2. Handle type-array + enum with null  (e.g. disc_secondary)
        #    and type-array without enum          (e.g. resolution)
        type_val = node.get("type")
        if isinstance(type_val, list) and None in node.get("enum", []):
            # Case: type array + enum with null
            non_null_types = [t for t in type_val if t != "null"]
            non_null_enums = [v for v in node["enum"] if v is not None]
            branches = []
            for t in non_null_types:
                branches.append({"type": t, "enum": non_null_enums})
            branches.append({"type": "null"})
            desc = node.get("description")
            node.clear()
            node["anyOf"] = branches
            if desc:
                node["description"] = desc
        elif isinstance(type_val, list) and "null" in type_val:
            # Case: type array without null in enum (e.g. ["string", "null"])
            non_null_types = [t for t in type_val if t != "null"]
            branches = [{"type": t} for t in non_null_types]
            branches.append({"type": "null"})
            del node["type"]
            node["anyOf"] = branches

        # 3. Recurse into sub-schemas
        for key in ("properties", "patternProperties"):
            if key in node and isinstance(node[key], dict):
                for prop_name, prop_schema in node[key].items():
                    node[key][prop_name] = _walk(prop_schema)
        if "items" in node and isinstance(node["items"], dict):
            node["items"] = _walk(node["items"])
        for key in ("anyOf", "allOf"):
            if key in node and isinstance(node[key], list):
                node[key] = [_walk(item) for item in node[key]]
        if "additionalProperties" in node and isinstance(node["additionalProperties"], dict):
            node["additionalProperties"] = _walk(node["additionalProperties"])
        return node

    return _walk(schema)


class ChatCompletionClient(Protocol):
    """Interface for chat completion providers."""

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        ...


class BasetenChatCompletionClient:
    """Baseten-backed chat completion client."""

    def __init__(self, *, base_url: str, default_model: str):
        self.base_url = base_url
        self.default_model = default_model
        self.api_key = load_env_value("BASETEN_API_KEY")

    @_retry_on_rate_limit
    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        temperature: float = 1,
        max_tokens: int = 100000,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("Baseten API key is required")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs: dict = dict(
            model=model or self.default_model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format
        completion = client.chat.completions.create(**kwargs)
        if completion and getattr(completion, "choices", None):
            first_choice = completion.choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) if message else ""
            if content:
                return str(content)
        print(completion)
        raise ValueError("LLM returned no content")


def _messages_to_contents(messages: Iterable[Mapping[str, str]]) -> list:
    """Convert OpenAI-style messages to Gemini contents."""
    contents = []
    for message in messages:
        role = str(message.get("role") or "user")
        text = str(message.get("content") or "").strip()
        if not text:
            continue
        # Gemini expects "user" or "model" roles; map system to user, assistant to model.
        mapped_role = "user" if role in ("system", "user") else "model"
        contents.append({"role": mapped_role, "parts": [{"text": text}]})
    return contents


class GeminiChatCompletionClient:
    """Gemini 3 chat completion client."""

    def __init__(self, *, default_model: str, api_version: str = "v1beta", thinking_level: str = "low", thinking_budget: Optional[int] = None):
        self.default_model = default_model
        self.api_version = api_version
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self.api_key = load_env_value("GEMINI_API_KEY")

    @_retry_on_rate_limit
    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai package is required for Gemini client") from exc

        contents = _messages_to_contents(messages)
        if not contents:
            raise ValueError("At least one message with text content is required")

        # Build Gemini config, translating OpenAI-style response_format
        thinking_kwargs: dict = {
            "thinking_level": types.ThinkingLevel[self.thinking_level.upper()],
        }
        if self.thinking_budget is not None:
            thinking_kwargs["thinking_budget"] = self.thinking_budget
        config_kwargs: dict = {
            "thinking_config": types.ThinkingConfig(**thinking_kwargs),
        }
        if response_format is not None:
            fmt_type = response_format.get("type")
            if fmt_type in ("json_object", "json_schema"):
                config_kwargs["response_mime_type"] = "application/json"
            if fmt_type == "json_schema":
                schema = response_format.get("json_schema", {}).get("schema")
                if schema:
                    config_kwargs["response_schema"] = schema

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=model or self.default_model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = getattr(response, "text", None)
        if text:
            return str(text)
        if getattr(response, "candidates", None):
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if part_text:
                            return str(part_text)
        print(response)
        raise ValueError("Gemini returned no content")


class VertexAIChatCompletionClient:
    """Kimi K2 Thinking via Vertex AI using the native Google GenAI SDK.

    Uses Application Default Credentials for Vertex AI auth.
    OTel tracing is set up once via otel_setup.setup_otel_tracing().
    """

    def __init__(
        self,
        *,
        default_model: str,
        project: Optional[str] = None,
        location: str = "global",
    ):
        self.default_model = default_model
        self.project = project or load_env_value("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location

    @_retry_on_rate_limit
    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai package is required for Vertex AI client") from exc

        from services.otel_setup import setup_otel_tracing
        setup_otel_tracing()

        client = genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
        )

        contents = _messages_to_contents(messages)
        if not contents:
            raise ValueError("At least one message with text content is required")

        config_kwargs: dict = {}
        if response_format is not None:
            fmt_type = response_format.get("type")
            if fmt_type in ("json_object", "json_schema"):
                config_kwargs["response_mime_type"] = "application/json"
            if fmt_type == "json_schema":
                schema = response_format.get("json_schema", {}).get("schema")
                if schema:
                    config_kwargs["response_schema"] = schema

        kwargs: dict = dict(
            model=model or self.default_model,
            contents=contents,
        )
        if config_kwargs:
            from google.genai import types
            kwargs["config"] = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(**kwargs)

        text = getattr(response, "text", None)
        if text:
            return str(text)
        if getattr(response, "candidates", None):
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if parts:
                    for part in parts:
                        if getattr(part, "thought", False):
                            continue
                        part_text = getattr(part, "text", None)
                        if part_text:
                            return str(part_text)
        print(response)
        raise ValueError("Vertex AI returned no content")


class FireworksChatCompletionClient:
    """Fireworks AI-backed chat completion client (OpenAI-compatible API)."""

    BASE_URL = "https://api.fireworks.ai/inference/v1"

    def __init__(
        self,
        *,
        default_model: str,
        temperature: float = 0.6,
        max_tokens: int = 4000,
        top_p: float = 1,
        top_k: int = 40,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        thinking_budget: Optional[int] = None,
    ):
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.thinking_budget = thinking_budget
        self.api_key = load_env_value("FIREWORKS_API_KEY")

    @_retry_on_rate_limit
    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("Fireworks API key is required")
        client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)
        extra_body: dict = {"top_k": self.top_k}
        if self.thinking_budget is not None:
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        kwargs: dict = dict(
            model=model or self.default_model,
            messages=list(messages),
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            extra_body=extra_body,
        )
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if response_format is not None:
            if response_format.get("type") == "json_schema":
                response_format = copy.deepcopy(response_format)
                inner = response_format.get("json_schema", {})
                if "schema" in inner:
                    inner["schema"] = _sanitize_schema_for_fireworks(inner["schema"])
            kwargs["response_format"] = response_format
        completion = client.chat.completions.create(**kwargs)
        if completion and getattr(completion, "choices", None):
            first_choice = completion.choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) if message else ""
            if content:
                return str(content)
        print(completion)
        raise ValueError("Fireworks returned no content")


class GroqChatCompletionClient:
    """Groq-backed chat completion client (OpenAI-compatible API)."""

    def __init__(
        self,
        *,
        default_model: str,
        temperature: float = 1,
        max_completion_tokens: int = 8192,
        top_p: float = 1,
        reasoning_effort: Optional[str] = None,
    ):
        self.default_model = default_model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.api_key = load_env_value("GROQ_API_KEY")

    @_retry_on_rate_limit
    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("Groq API key is required")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("groq package is required for Groq client") from exc

        client = Groq(api_key=self.api_key)
        kwargs: dict = dict(
            model=model or self.default_model,
            messages=list(messages),
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            top_p=self.top_p,
        )
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if response_format is not None:
            kwargs["response_format"] = response_format

        completion = client.chat.completions.create(**kwargs)
        if completion and getattr(completion, "choices", None):
            first_choice = completion.choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) if message else ""
            if content:
                return str(content)
        print(completion)
        raise ValueError("Groq returned no content")


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict = {
    "gemini": {"default_model": "gemini-3-flash-preview", "thinking_level": "low"},
    "gemini-lite": {"default_model": "gemini-3-flash-preview", "thinking_level": "low", "thinking_budget": 1024},
    "groq": {"default_model": "openai/gpt-oss-120b", "reasoning_effort": "medium"},
    "fireworks": {"default_model": "accounts/fireworks/models/kimi-k2-thinking"},
    "fireworks-lite": {"default_model": "accounts/fireworks/models/kimi-k2-thinking", "thinking_budget": 1024},
    "vertex": {"default_model": "moonshotai/kimi-k2-thinking-maas", "location": "global"},
    "baseten": {},  # requires base_url + default_model at call site
}

_PROVIDER_CLASSES: dict = {
    "gemini": GeminiChatCompletionClient,
    "gemini-lite": GeminiChatCompletionClient,
    "groq": GroqChatCompletionClient,
    "fireworks": FireworksChatCompletionClient,
    "fireworks-lite": FireworksChatCompletionClient,
    "vertex": VertexAIChatCompletionClient,
    "baseten": BasetenChatCompletionClient,
}


def get_llm_client(provider: str, **kwargs) -> ChatCompletionClient:
    """Create an LLM client by provider name, merging caller overrides with defaults."""
    if provider not in _PROVIDER_CLASSES:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDER_CLASSES)}")
    merged = {**_PROVIDER_DEFAULTS.get(provider, {}), **kwargs}
    return _PROVIDER_CLASSES[provider](**merged)


# ---------------------------------------------------------------------------
# Pre-built singletons
# ---------------------------------------------------------------------------

GEMINI_3_PRO_MODEL = "gemini-3-flash-preview"

# Optional Gemini-backed client; uses low thinking level by default.
scenario_selector_gemini_llm: ChatCompletionClient = GeminiChatCompletionClient(
    default_model=GEMINI_3_PRO_MODEL,
    thinking_level="low",
)
