from unittest.mock import MagicMock
from services.supabase_client import SupabaseService


def make_mock_service():
    """Create a SupabaseService with a mocked client."""
    service = SupabaseService.__new__(SupabaseService)
    service.client = MagicMock()
    return service


def test_upsert_competitors_calls_delete_then_insert():
    """upsert_competitors deletes old rows then inserts new ones."""
    service = make_mock_service()

    table_mock = MagicMock()
    service.client.table.return_value = table_mock
    table_mock.delete.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.insert.return_value = table_mock
    table_mock.execute.return_value = MagicMock(data=[])

    competitors = [
        {"competitor_id": "norlys", "name": "Norlys", "aliases": ["Eniig"],
         "research_data": {"overview": "test"}},
    ]
    count = service.upsert_competitors("campaign_123", competitors)

    assert count == 1
    assert service.client.table.call_count >= 2


def test_get_competitor_research_data():
    """get_competitor_research_data returns the jsonb dict."""
    service = make_mock_service()

    table_mock = MagicMock()
    service.client.table.return_value = table_mock
    table_mock.select.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.single.return_value = table_mock
    table_mock.execute.return_value = MagicMock(
        data={"research_data": {"overview": "Big company", "pricing_packages": []}}
    )

    result = service.get_competitor_research_data("campaign_123", "norlys")
    assert result is not None
    assert result["overview"] == "Big company"
