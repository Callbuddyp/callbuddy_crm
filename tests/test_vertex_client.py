# tests/test_vertex_client.py
from unittest.mock import patch, MagicMock


def test_vertex_client_generate_returns_text(monkeypatch):
    """VertexAIChatCompletionClient.generate() returns the text from the response."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    mock_response = MagicMock()
    mock_response.text = "Hello from Kimi K2"
    mock_response.candidates = None

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance

    # The google mock's genai attribute must point to our mock_genai
    # so that `from google import genai` resolves correctly.
    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
        import importlib
        import services.llm_client as llm_mod
        importlib.reload(llm_mod)

        client = llm_mod.VertexAIChatCompletionClient(default_model="kimi-k2-thinking-maas")
        result = client.generate(messages=[{"role": "user", "content": "Hello"}])

    assert result == "Hello from Kimi K2"


def test_vertex_client_falls_back_to_candidates_skipping_thoughts(monkeypatch):
    """When response.text is None, extract text from candidates, skipping thought parts."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    thought_part = MagicMock()
    thought_part.text = "Let me think about this..."
    thought_part.thought = True

    answer_part = MagicMock()
    answer_part.text = "Fallback text"
    answer_part.thought = False

    mock_content = MagicMock()
    mock_content.parts = [thought_part, answer_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response = MagicMock()
    mock_response.text = None
    mock_response.candidates = [mock_candidate]

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client_instance

    mock_google = MagicMock()
    mock_google.genai = mock_genai

    with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
        import importlib
        import services.llm_client as llm_mod
        importlib.reload(llm_mod)

        client = llm_mod.VertexAIChatCompletionClient(default_model="kimi-k2-thinking-maas")
        result = client.generate(messages=[{"role": "user", "content": "Hello"}])

    assert result == "Fallback text"


def test_get_llm_client_vertex_returns_vertex_client(monkeypatch):
    """get_llm_client('vertex') should return a VertexAIChatCompletionClient."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    from services.llm_client import get_llm_client, VertexAIChatCompletionClient

    client = get_llm_client("vertex")
    assert isinstance(client, VertexAIChatCompletionClient)
    assert client.default_model == "moonshotai/kimi-k2-thinking-maas"
    assert client.location == "global"
