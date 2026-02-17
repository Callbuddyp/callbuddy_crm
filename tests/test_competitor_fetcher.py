import json
from unittest.mock import MagicMock, patch

from services.competitor_fetcher import (
    render_competitor_briefs,
    build_research_prompt,
    build_evaluator_prompt,
    format_research_results_md,
    render_competitor_description,
    normalize_competitor,
    _discover_competitors_gemini,
    _call_gemini_with_grounding,
)


SAMPLE_COMPETITORS = [
    {
        "name": "Norlys",
        "aliases": ["Eniig", "Stofa"],
        "rationale": "Largest energy company",
        "discovered_urls": [
            "https://norlys.dk/el/produkter",
            "https://dk.trustpilot.com/review/norlys.dk",
        ],
        "focus_areas": "Recently raised spot-tillaeg by 3 oere/kWh.",
    },
    {
        "name": "Andel Energi",
        "aliases": ["SEAS-NVE"],
        "rationale": "Eastern Denmark cooperative",
        "discovered_urls": ["https://andelenergi.dk/el"],
        "focus_areas": "Competitive spot prices, no binding.",
    },
]


def test_render_competitor_briefs_format():
    """render_competitor_briefs produces readable markdown with URLs and focus areas."""
    result = render_competitor_briefs(SAMPLE_COMPETITORS)
    assert "## Norlys" in result
    assert "(also known as: Eniig, Stofa)" in result
    assert "https://norlys.dk/el/produkter" in result
    assert "Recently raised spot-tillaeg" in result
    assert "## Andel Energi" in result
    assert "(also known as: SEAS-NVE)" in result


def test_render_competitor_briefs_empty():
    """render_competitor_briefs handles empty list."""
    assert render_competitor_briefs([]) == ""


def test_render_competitor_briefs_no_aliases():
    """render_competitor_briefs omits alias line when empty."""
    comps = [{"name": "Barry", "aliases": [], "rationale": "Spot app",
              "discovered_urls": [], "focus_areas": "App-only provider."}]
    result = render_competitor_briefs(comps)
    assert "## Barry" in result
    assert "also known as" not in result


def test_build_research_prompt_injects_variables():
    """build_research_prompt replaces template variables."""
    template = "Campaign: {{CAMPAIGN_INFO}}\nMessages: {{CUSTOM_MESSAGES}}\nBriefs: {{COMPETITOR_BRIEFS}}"
    result = build_research_prompt(
        template=template,
        campaign_info="ELG B2C info",
        custom_messages="Focus on binding",
        competitor_briefs="## Norlys\nDetails here",
    )
    assert "ELG B2C info" in result
    assert "Focus on binding" in result
    assert "## Norlys" in result


def test_build_research_prompt_empty_optionals():
    """build_research_prompt handles empty optional variables gracefully."""
    template = "Info: {{CAMPAIGN_INFO}}\nMsgs: {{CUSTOM_MESSAGES}}\nBriefs: {{COMPETITOR_BRIEFS}}"
    result = build_research_prompt(
        template=template,
        campaign_info="Test info",
        custom_messages="",
        competitor_briefs="## Test",
    )
    assert "Msgs: " in result


def test_render_competitor_description_produces_readable_text():
    """render_competitor_description turns research_data into tool-friendly text."""
    data = {
        "overview": "Largest energy company in Denmark.",
        "pricing_packages": [
            {"name": "Basis", "price": "49 kr/md", "details": "No binding", "comparison_to_us": "10 kr dyrere"}
        ],
        "pain_points": [
            {"claim": "Impossible to cancel", "story": "A customer tried 3 times..."}
        ],
        "critical_news": [
            {"date": "2025-11", "headline": "Price hike", "summary": "Raised prices 15%"}
        ],
    }
    result = render_competitor_description(data)
    assert "Largest energy company" in result
    assert "49 kr/md" in result
    assert "Impossible to cancel" in result
    assert "Price hike" in result


def test_format_research_results_md():
    """format_research_results_md renders full report for CLI display."""
    output = {
        "market_overview": "Danish electricity market overview.",
        "competitors": [
            {
                "name": "Norlys",
                "overview": "Big company",
                "pricing_packages": [],
                "pain_points": [{"claim": "Bad", "story": "Very bad"}],
                "critical_news": [],
            }
        ],
    }
    result = format_research_results_md(output)
    assert "## Market Overview" in result
    assert "### Norlys" in result
    assert "Pain Points (1)" in result


def test_discover_competitors_parses_json_response():
    """_discover_competitors_gemini parses Gemini grounding response."""
    mock_response = MagicMock()
    mock_response.text = json.dumps([
        {
            "name": "Norlys",
            "aliases": ["Eniig"],
            "rationale": "Largest energy company",
            "discovered_urls": ["https://norlys.dk"],
            "focus_areas": "Bundles gas+el",
        }
    ])

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("services.competitor_fetcher.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = _discover_competitors_gemini(
            api_key="test-key",
            prompt="Find competitors",
        )

    assert len(result) == 1
    assert result[0]["name"] == "Norlys"
    assert result[0]["aliases"] == ["Eniig"]


def test_discover_competitors_strips_code_fences():
    """_discover_competitors_gemini handles markdown-wrapped JSON."""
    mock_response = MagicMock()
    mock_response.text = '```json\n[{"name": "Barry", "aliases": [], "rationale": "Spot app", "discovered_urls": [], "focus_areas": ""}]\n```'

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("services.competitor_fetcher.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = _discover_competitors_gemini(api_key="k", prompt="p")

    assert result[0]["name"] == "Barry"


SAMPLE_V2_COMPETITOR = {
    "name": "Norlys",
    "competitor_id": "norlys",
    "aliases": ["Eniig", "SE"],
    "comparison_summary": "Norlys tager 29 kr/md og 9 øre/kWh tillæg. Vi tager 19 kr/md og 4 øre/kWh.",
    "strengths": [
        "Stærk lokal forankring som andelsejet selskab.",
        "God app med live timepris-oversigt.",
    ],
    "angles": [
        "Norlys' FlexEl har 9 øre/kWh vs vores 4 øre — ca. 200 kr/år dyrere.",
        "Ingen velkomstrabat hos Norlys — vi tilbyder 3 gratis måneder.",
    ],
    "switching_barriers": [
        "Ingen binding, men mange tror fejlagtigt de er bundet.",
    ],
    "anecdotes": [
        "En kunde ventede 45 minutter i telefon efter et estimat på 11 minutter.",
    ],
    "recent_news": [
        "Norlys hævede FlexEl-tillægget fra 7 til 9 øre i oktober 2025.",
    ],
}


def test_render_competitor_description_v2():
    """render_competitor_description handles the v2 schema with all new fields."""
    result = render_competitor_description(SAMPLE_V2_COMPETITOR)
    assert "Norlys" in result
    assert "Eniig, SE" in result
    assert "29 kr/md" in result
    assert "Stærk lokal forankring" in result
    assert "9 øre/kWh vs vores 4 øre" in result
    assert "Ingen binding" in result
    assert "45 minutter" in result
    assert "oktober 2025" in result


def test_render_competitor_description_v2_empty_optional_fields():
    """render_competitor_description handles empty arrays gracefully."""
    data = {
        "name": "TestCorp",
        "competitor_id": "testcorp",
        "aliases": [],
        "comparison_summary": "They are more expensive.",
        "strengths": [],
        "angles": ["We are cheaper."],
        "switching_barriers": [],
        "anecdotes": [],
        "recent_news": [],
    }
    result = render_competitor_description(data)
    assert "TestCorp" in result
    assert "They are more expensive." in result
    assert "We are cheaper." in result
    # Section headers should be omitted when arrays are empty
    assert "De er gode til" not in result
    assert "Skiftbarrierer" not in result
    assert "Historier" not in result
    assert "Seneste nyt" not in result


def test_render_competitor_description_v1_backwards_compat():
    """render_competitor_description still works with v1 research_data format."""
    data = {
        "overview": "Largest energy company in Denmark.",
        "pricing_packages": [
            {"name": "Basis", "price": "49 kr/md", "details": "No binding", "comparison_to_us": "10 kr dyrere"}
        ],
        "pain_points": [
            {"claim": "Impossible to cancel", "story": "A customer tried 3 times..."}
        ],
        "critical_news": [
            {"date": "2025-11", "headline": "Price hike", "summary": "Raised prices 15%"}
        ],
    }
    result = render_competitor_description(data)
    assert "Largest energy company" in result
    assert "49 kr/md" in result
    assert "Impossible to cancel" in result
    assert "Price hike" in result


def test_build_evaluator_prompt_injects_variables():
    """build_evaluator_prompt replaces template variables."""
    template = "Campaign: {{CAMPAIGN_INFO}}\nResearcher: {{RESEARCHER_OUTPUT}}"
    result = build_evaluator_prompt(
        template=template,
        campaign_info="ELG B2C info",
        researcher_output='[{"name": "Norlys"}]',
    )
    assert "ELG B2C info" in result
    assert "Norlys" in result


def test_format_research_results_md_v2():
    """format_research_results_md handles v2 competitor list in dict wrapper."""
    output = {"competitors": [SAMPLE_V2_COMPETITOR]}
    result = format_research_results_md(output)
    assert "Norlys" in result
    assert "9 øre/kWh vs vores 4 øre" in result


def test_format_research_results_md_v2_raw_list():
    """format_research_results_md handles v2 competitor list passed directly."""
    result = format_research_results_md([SAMPLE_V2_COMPETITOR])
    assert "Norlys" in result
    assert "Vores fordele" in result


def test_call_gemini_with_grounding_parses_json():
    """_call_gemini_with_grounding parses Gemini grounding response."""
    mock_response = MagicMock()
    mock_response.text = json.dumps([{
        "name": "Norlys",
        "competitor_id": "norlys",
        "aliases": ["Eniig"],
        "comparison_summary": "Test comparison",
        "strengths": ["Strong brand"],
        "angles": ["Our price is lower"],
        "switching_barriers": ["No binding"],
        "anecdotes": ["A customer waited 45 min"],
        "recent_news": ["Price hike Oct 2025"],
    }])

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("services.competitor_fetcher.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        result = _call_gemini_with_grounding(
            api_key="test-key",
            prompt="Research competitors",
        )

    assert len(result) == 1
    assert result[0]["name"] == "Norlys"
    assert result[0]["angles"] == ["Our price is lower"]


# ---------------------------------------------------------------------------
# normalize_competitor tests
# ---------------------------------------------------------------------------

def test_normalize_competitor_flattens_identification():
    """normalize_competitor extracts name/id/aliases from nested identification."""
    comp = {
        "identification": {
            "name": "Norlys",
            "competitor_id": "norlys",
            "aliases": ["Eniig", "SE"],
        },
        "comparison_summary": "flat string",
        "strengths": [],
        "angles": [],
        "switching_barriers": [],
        "anecdotes": [],
        "recent_news": [],
    }
    result = normalize_competitor(comp)
    assert result["name"] == "Norlys"
    assert result["competitor_id"] == "norlys"
    assert result["aliases"] == ["Eniig", "SE"]


def test_normalize_competitor_flattens_comparison_summary_dict():
    """normalize_competitor joins comparison_summary sub-keys into one string."""
    comp = {
        "name": "Norlys",
        "competitor_id": "norlys",
        "aliases": [],
        "comparison_summary": {
            "spot_price": "Spot costs 9 øre/kWh.",
            "fixed_price": "Fixed costs 29 kr/md.",
            "general_terms": "No binding.",
        },
        "strengths": [],
        "angles": [],
        "switching_barriers": [],
        "anecdotes": [],
        "recent_news": [],
    }
    result = normalize_competitor(comp)
    assert isinstance(result["comparison_summary"], str)
    assert "9 øre/kWh" in result["comparison_summary"]
    assert "29 kr/md" in result["comparison_summary"]
    assert "No binding" in result["comparison_summary"]


def test_normalize_competitor_flattens_switching_barriers_dict():
    """normalize_competitor flattens switching_barriers from grouped dict to flat list."""
    comp = {
        "name": "Norlys",
        "competitor_id": "norlys",
        "aliases": [],
        "comparison_summary": "",
        "strengths": [],
        "angles": [],
        "switching_barriers": {
            "contractual": ["6 month binding on TV."],
            "practical": ["Bundled with broadband."],
            "psychological": ["Strong local brand.", "Habit."],
        },
        "anecdotes": [],
        "recent_news": [],
    }
    result = normalize_competitor(comp)
    assert isinstance(result["switching_barriers"], list)
    assert len(result["switching_barriers"]) == 4
    assert "6 month binding" in result["switching_barriers"][0]
    assert "Habit." in result["switching_barriers"][3]


def test_normalize_competitor_flattens_recent_news_dicts():
    """normalize_competitor converts recent_news dicts to '[date] text' strings."""
    comp = {
        "name": "Norlys",
        "competitor_id": "norlys",
        "aliases": [],
        "comparison_summary": "",
        "strengths": [],
        "angles": [],
        "switching_barriers": [],
        "anecdotes": [],
        "recent_news": [
            {"date": "Oktober 2025", "news_item": "Price hike 15%."},
            {"date": "Februar 2026", "headline": "New CEO."},
            "Already flat string.",
        ],
    }
    result = normalize_competitor(comp)
    assert result["recent_news"][0] == "[Oktober 2025] Price hike 15%."
    assert result["recent_news"][1] == "[Februar 2026] New CEO."
    assert result["recent_news"][2] == "Already flat string."


def test_normalize_competitor_generates_competitor_id():
    """normalize_competitor generates competitor_id from name when missing."""
    comp = {
        "name": "Andel Energi",
        "aliases": [],
        "comparison_summary": "",
        "strengths": [],
        "angles": [],
        "switching_barriers": [],
        "anecdotes": [],
        "recent_news": [],
    }
    result = normalize_competitor(comp)
    assert result["competitor_id"] == "andel_energi"
