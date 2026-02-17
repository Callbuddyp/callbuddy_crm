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
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
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
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set as global provider so the instrumentor picks it up.
    from opentelemetry import trace
    try:
        trace.set_tracer_provider(provider)
    except RuntimeError:
        # Provider already set (e.g. by another module) — reuse it.
        pass

    GoogleGenAiSdkInstrumentor().instrument()
    _initialised = True
    print(f"[otel] Tracing enabled — exporting to {host}")
