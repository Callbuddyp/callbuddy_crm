# Vertex AI Kimi K2 Thinking Client Design

## Goal

Add Kimi K2 Thinking as a new LLM provider via Vertex AI's native Google GenAI SDK, with automatic OpenTelemetry tracing exported to Langfuse.

## Architecture

### Components

1. **`VertexAIChatCompletionClient`** in `src/services/llm_client.py`
   - Uses `google.genai.Client(vertexai=True, project=..., location="global")`
   - Calls `client.models.generate_content(model="kimi-k2-thinking-maas", ...)`
   - Implements the existing `ChatCompletionClient` protocol
   - Converts OpenAI-style messages to GenAI contents (reuses `_messages_to_contents`)
   - Extracts final text from response, skipping thinking parts

2. **OpenTelemetry instrumentation setup** (new module or inline)
   - `GoogleGenAiSdkInstrumentor().instrument()` auto-instruments all `generate_content` calls
   - OTLP HTTP exporter sends traces to Langfuse's `/api/public/otel` endpoint
   - Auth via Basic Auth using existing `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`
   - Capture message content enabled via `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`

3. **Provider registration** in the factory
   - Provider name: `"vertex"`
   - Default model: `"kimi-k2-thinking-maas"`
   - Registered in `_PROVIDER_DEFAULTS` and `_PROVIDER_CLASSES`

### Authentication

- Vertex AI: Application Default Credentials (ADC) via `gcloud auth application-default login`
- Langfuse OTLP: Basic Auth from existing env vars (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)

### Data flow

```
Script calls get_llm_client("vertex")
  -> VertexAIChatCompletionClient.generate()
    -> genai.Client.models.generate_content("kimi-k2-thinking-maas", ...)
      -> OTel instrumentor captures span (input, output, timing, tokens)
        -> OTLP exporter -> Langfuse /api/public/otel
    -> Extract text from response -> return to caller
```

### Environment variables

| Variable | Purpose | New? |
|----------|---------|------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for Vertex AI | Yes |
| `LANGFUSE_PUBLIC_KEY` | Langfuse auth (already exists) | No |
| `LANGFUSE_SECRET_KEY` | Langfuse auth (already exists) | No |
| `LANGFUSE_HOST` | Langfuse host (already exists) | No |

### New dependencies

- `opentelemetry-instrumentation-google-genai` (0.6b0)
- `opentelemetry-exporter-otlp-proto-http`
- `opentelemetry-sdk`

`google-genai` is already installed (used by `GeminiChatCompletionClient`).

### What Langfuse will capture automatically

- Request/response content (inputs, outputs, reasoning trace)
- Latency (span duration)
- Token counts (input/output)
- Model name and parameters
- Errors
