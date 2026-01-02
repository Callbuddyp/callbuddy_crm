from __future__ import annotations

from typing import Iterable, Mapping, Optional, Protocol, Sequence

from openai import OpenAI
from utils import load_env_value

class ChatCompletionClient(Protocol):
    """Interface for chat completion providers."""

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
    ) -> str:
        ...


class BasetenChatCompletionClient:
    """Baseten-backed chat completion client."""

    def __init__(self, *, base_url: str, default_model: str):
        self.base_url = base_url
        self.default_model = default_model
        self.api_key = load_env_value("BASETEN_API_KEY")
    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
        temperature: float = 1,
        max_tokens: int = 100000,
    ) -> str:
        if not self.api_key:
            raise ValueError("Baseten API key is required")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        completion = client.chat.completions.create(
            model=model or self.default_model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
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

    def __init__(self, *, default_model: str, api_version: str = "v1beta", thinking_level: str = "low"):
        self.default_model = default_model
        self.api_version = api_version
        self.thinking_level = thinking_level
        self.api_key = load_env_value("GEMINI_API_KEY")

    def generate(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: Optional[str] = None,
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

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=model or self.default_model,
            contents=contents,
                config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            ),
            
           
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


GEMINI_3_PRO_MODEL = "gemini-3-flash-preview"

# Optional Gemini-backed client; uses low thinking level by default.
scenario_selector_gemini_llm: ChatCompletionClient = GeminiChatCompletionClient(
    default_model=GEMINI_3_PRO_MODEL,
    thinking_level="low",
)
