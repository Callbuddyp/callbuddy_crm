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

        import importlib
        import services.otel_setup as otel_mod
        importlib.reload(otel_mod)

        otel_mod.setup_otel_tracing()
        mock_instrumentor.instrument.assert_called_once()


def test_setup_otel_tracing_noop_without_langfuse_keys(monkeypatch):
    """If Langfuse keys are missing, setup_otel_tracing should not crash."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    import importlib
    import services.otel_setup as otel_mod
    importlib.reload(otel_mod)

    otel_mod.setup_otel_tracing()
