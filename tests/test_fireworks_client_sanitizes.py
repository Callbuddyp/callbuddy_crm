# tests/test_fireworks_client_sanitizes.py
"""Verify FireworksChatCompletionClient applies schema sanitization."""
from unittest.mock import patch, MagicMock


def test_fireworks_client_sanitizes_schema_before_sending(monkeypatch):
    """When response_format has json_schema, the schema sent to OpenAI must be sanitized."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

    from services.llm_client import FireworksChatCompletionClient

    mock_choice = MagicMock()
    mock_choice.message.content = '{"patches": []}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("services.llm_client.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        MockOpenAI.return_value = mock_client

        client = FireworksChatCompletionClient(default_model="test-model")
        client.generate(
            messages=[{"role": "user", "content": "test"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                            }
                        },
                    },
                },
            },
        )

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    sent_schema = call_kwargs["response_format"]["json_schema"]["schema"]
    # oneOf should have been rewritten to anyOf
    assert "oneOf" not in sent_schema["properties"]["items"]["items"]
    assert "anyOf" in sent_schema["properties"]["items"]["items"]
