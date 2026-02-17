# Vertex AI Kimi K2 Thinking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `VertexAIChatCompletionClient` that calls Kimi K2 Thinking via the native Google GenAI SDK on Vertex AI, with automatic OpenTelemetry tracing exported to Langfuse.

**Architecture:** New client class in `llm_client.py` using `google.genai.Client(vertexai=True)` and `generate_content()`. A small `otel_setup.py` module initializes the `GoogleGenAiSdkInstrumentor` and configures an OTLP HTTP exporter pointed at Langfuse. The client is registered in the existing provider factory as `"vertex"`.

**Tech Stack:** `google-genai` (already installed), `opentelemetry-instrumentation-google-genai`, `opentelemetry-exporter-otlp-proto-http`, `opentelemetry-sdk`

---

### Task 1: Install new dependencies

**Files:**
- Modify: `src/requirements.txt`

**Step 1: Add OTel dependencies to requirements.txt**

Add these lines to `src/requirements.txt`:

```
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp-proto-http>=1.20.0
opentelemetry-instrumentation-google-genai>=0.6b0
```

**Step 2: Install**

Run: `pip install -r src/requirements.txt`
Expected: All packages install successfully.

**Step 3: Commit**

```bash
git add src/requirements.txt
git commit -m "feat: add OpenTelemetry dependencies for Vertex AI tracing"
```

---

### Task 2: Create OpenTelemetry setup module

**Files:**
- Create: `src/services/otel_setup.py`
- Test: `tests/test_otel_setup.py`

**Step 1: Write the failing test**

```python
# tests/test_otel_setup.py
from unittest.mock import patch, MagicMock


def test_setup_otel_tracing_instruments_genai(monkeypatch):
    """Verify that setup_otel_tracing calls GoogleGenAiSdkInstrumentor.instrument()."""
    mock_instrumentor_cls = MagicMock()
    mock_instrumentor = MagicMock()
    mock_instrumentor_cls.return_value = mock_instrumentor

    with patch.dict("sys.modules", {
        "opentelemetry.instrumentation.google_genai": MagicMock(
            GoogleGenAiSdkInstrumentor=mock_instrumentor_cls
        ),
        "opentelemetry.sdk.trace": MagicMock(),
        "opentelemetry.sdk.trace.export": MagicMock(),
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": MagicMock(),
    }):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        # Re-import to pick up mocked modules
        import importlib
        import services.otel_setup as otel_mod
        importlib.reload(otel_mod)

        otel_mod.setup_otel_tracing()
        mock_instrumentor.instrument.assert_called_once()


def test_setup_otel_tracing_noop_without_langfuse_keys(monkeypatch):
    """If Langfuse keys are missing, setup_otel_tracing should not crash."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    from services.otel_setup import setup_otel_tracing
    # Should not raise
    setup_otel_tracing()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_otel_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.otel_setup'`

**Step 3: Write the implementation**

```python
# src/services/otel_setup.py
"""One-time OpenTelemetry setup: instruments the Google GenAI SDK and
exports traces to Langfuse via OTLP/HTTP."""

from __future__ import annotations

import base64
import os

from utils import load_env_value

_initialised = False


def setup_otel_tracing() -> None:
    """Instrument the Google GenAI SDK and configure the OTLP exporter for Langfuse.

    Safe to call multiple times — only the first invocation has an effect.
    If Langfuse credentials are missing the function silently returns.
    """
    global _initialised
    if _initialised:
        return

    public_key = load_env_value("LANGFUSE_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = load_env_value("LANGFUSE_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY")
    host = load_env_value("LANGFUSE_HOST") or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"

    if not public_key or not secret_key:
        print("[otel] Langfuse keys not found — skipping OTel setup.")
        return

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
    except ImportError as exc:
        print(f"[otel] OTel packages not installed — skipping: {exc}")
        return

    # Enable content capture (inputs/outputs/reasoning).
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    # Build Basic-auth header for Langfuse OTLP endpoint.
    auth_bytes = f"{public_key}:{secret_key}".encode()
    auth_header = base64.b64encode(auth_bytes).decode()

    endpoint = f"{host.rstrip('/')}/api/public/otel/v1/traces"
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={"Authorization": f"Basic {auth_header}"},
    )

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Set as global provider so the instrumentor picks it up.
    from opentelemetry import trace
    trace.set_tracer_provider(provider)

    GoogleGenAiSdkInstrumentor().instrument()
    _initialised = True
    print(f"[otel] Tracing enabled — exporting to {host}")
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_otel_setup.py -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/services/otel_setup.py tests/test_otel_setup.py
git commit -m "feat: add OTel setup module for GenAI SDK tracing to Langfuse"
```

---

### Task 3: Add VertexAIChatCompletionClient

**Files:**
- Modify: `src/services/llm_client.py`
- Test: `tests/test_vertex_client.py`

**Step 1: Write the failing test**

```python
# tests/test_vertex_client.py
from unittest.mock import patch, MagicMock


def test_vertex_client_generate_returns_text(monkeypatch):
    """VertexAIChatCompletionClient.generate() returns the text from the response."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    # Mock the genai module
    mock_response = MagicMock()
    mock_response.text = "Hello from Kimi K2"
    mock_response.candidates = None

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance

    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
        from services.llm_client import VertexAIChatCompletionClient

        client = VertexAIChatCompletionClient(default_model="kimi-k2-thinking-maas")
        result = client.generate(messages=[{"role": "user", "content": "Hello"}])

    assert result == "Hello from Kimi K2"


def test_vertex_client_falls_back_to_candidates(monkeypatch):
    """When response.text is None, extract text from candidates."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    mock_part = MagicMock()
    mock_part.text = "Fallback text"
    mock_part.thought = False

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = [mock_candidate]

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance

    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
        from services.llm_client import VertexAIChatCompletionClient

        client = VertexAIChatCompletionClient(default_model="kimi-k2-thinking-maas")
        result = client.generate(messages=[{"role": "user", "content": "Hello"}])

    assert result == "Fallback text"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_vertex_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'VertexAIChatCompletionClient'`

**Step 3: Write the implementation**

Add the following class to `src/services/llm_client.py` after the `GeminiChatCompletionClient` class (around line 168):

```python
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
```

Also add `import os` to the imports at the top of the file (line 1 area).

**Step 4: Run test to verify it passes**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_vertex_client.py -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/services/llm_client.py tests/test_vertex_client.py
git commit -m "feat: add VertexAIChatCompletionClient for Kimi K2 Thinking"
```

---

### Task 4: Register the Vertex AI provider in the factory

**Files:**
- Modify: `src/services/llm_client.py` (lines ~292-313)

**Step 1: Write the failing test**

```python
# tests/test_vertex_factory.py
from unittest.mock import patch, MagicMock


def test_get_llm_client_vertex_returns_vertex_client(monkeypatch):
    """get_llm_client('vertex') should return a VertexAIChatCompletionClient."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    from services.llm_client import get_llm_client, VertexAIChatCompletionClient

    client = get_llm_client("vertex")
    assert isinstance(client, VertexAIChatCompletionClient)
    assert client.default_model == "kimi-k2-thinking-maas"
    assert client.location == "global"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_vertex_factory.py -v`
Expected: FAIL — `ValueError: Unknown provider 'vertex'`

**Step 3: Add Vertex AI to the provider registry**

In `src/services/llm_client.py`, update `_PROVIDER_DEFAULTS` and `_PROVIDER_CLASSES`:

Add to `_PROVIDER_DEFAULTS` dict:
```python
    "vertex": {"default_model": "kimi-k2-thinking-maas", "location": "global"},
```

Add to `_PROVIDER_CLASSES` dict:
```python
    "vertex": VertexAIChatCompletionClient,
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_vertex_factory.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/services/llm_client.py tests/test_vertex_factory.py
git commit -m "feat: register vertex provider in LLM client factory"
```

---

### Task 5: End-to-end smoke test

**Files:**
- None (manual verification)

**Step 1: Run all tests**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/ -v`
Expected: All tests pass, no regressions.

**Step 2: Verify ADC is set up**

Run: `gcloud auth application-default print-access-token`
Expected: Prints a token (confirm you're logged in).

**Step 3: Quick manual smoke test**

Run from project root:
```bash
cd /Users/mikkeldahl/callbuddy_service/src
python -c "
from services.llm_client import get_llm_client
client = get_llm_client('vertex')
result = client.generate(messages=[{'role': 'user', 'content': 'Say hello in Danish'}])
print(result)
"
```
Expected: Prints a Danish greeting. Check Langfuse dashboard for a new trace with timing, tokens, and content.

**Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "test: verify Vertex AI Kimi K2 end-to-end"
```
