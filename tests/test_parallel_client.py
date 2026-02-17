from unittest.mock import MagicMock, patch

from services.parallel_client import ParallelClient


def test_deep_research_sends_correct_params():
    """ParallelClient.deep_research calls the SDK with correct processor and schema."""
    mock_client = MagicMock()
    mock_client.task_run.create.return_value = MagicMock(run_id="run_123")
    mock_client.task_run.result.return_value = MagicMock(
        output={"market_overview": "Test overview", "competitors": []}
    )

    with patch("services.parallel_client.Parallel", return_value=mock_client):
        pc = ParallelClient(api_key="test-key")
        schema = {"type": "object", "properties": {"market_overview": {"type": "string"}}}
        result = pc.deep_research(
            prompt="Research competitors",
            processor="pro",
            output_schema=schema,
        )

    # Verify SDK was called correctly
    create_call = mock_client.task_run.create.call_args
    assert create_call.kwargs["input"] == "Research competitors"
    assert create_call.kwargs["processor"] == "pro"
    assert create_call.kwargs["task_spec"]["output_schema"]["type"] == "json"
    assert create_call.kwargs["task_spec"]["output_schema"]["json_schema"] == schema

    # Verify result polling
    mock_client.task_run.result.assert_called_once_with("run_123", api_timeout=3600)
    assert result == {"market_overview": "Test overview", "competitors": []}


def test_deep_research_default_processor():
    """ParallelClient.deep_research defaults to 'pro' processor."""
    mock_client = MagicMock()
    mock_client.task_run.create.return_value = MagicMock(run_id="run_456")
    mock_client.task_run.result.return_value = MagicMock(output={})

    with patch("services.parallel_client.Parallel", return_value=mock_client):
        pc = ParallelClient(api_key="test-key")
        pc.deep_research(prompt="Test")

    assert mock_client.task_run.create.call_args.kwargs["processor"] == "pro"
