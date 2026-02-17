# Competitor Intelligence Fetcher — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reusable `CompetitorFetcherService` that discovers competitors via Gemini grounding, deep-researches them via Parallel AI, and stores structured results in Supabase — plus a thin CLI wrapper for interactive use.

**Architecture:** The service class `CompetitorFetcherService` exposes `generate_competitor_knowledge()` as the main entry point. Internally it runs a two-phase pipeline (Gemini discovery → Parallel AI deep research → Supabase storage). The CLI script (`run_competitor_fetcher.py`) is a thin interactive wrapper that collects user input and calls the service. This separation lets other code (onboarding, API endpoints) use the service directly without the CLI.

**Tech Stack:** Python 3.12, `google-genai` SDK (Gemini + grounding), `parallel` SDK (Parallel AI Task API), `supabase-py`, Langfuse SDK. No new frameworks.

**Design doc:** `docs/plans/2026-02-12-competitor-fetcher-design.md`

---

### Task 1: Install Parallel AI SDK and set up test infrastructure

**Files:**
- Modify: `src/requirements.txt`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

**Step 1: Add parallel SDK to requirements**

Add to `src/requirements.txt`:
```
parallel>=0.1.0
```

**Step 2: Install the dependency**

Run: `pip install parallel`

**Step 3: Create pytest config**

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = src
```

**Step 4: Create test conftest**

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

# Ensure src/ is on path for bare imports
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
```

**Step 5: Verify pytest runs**

Run: `pytest --co -q`
Expected: "no tests ran" (no error)

**Step 6: Commit**

```bash
git add src/requirements.txt tests/conftest.py pytest.ini
git commit -m "chore: add parallel SDK and pytest infrastructure"
```

---

### Task 2: Create ParallelClient wrapper

**Files:**
- Create: `src/services/parallel_client.py`
- Create: `tests/test_parallel_client.py`

**Step 1: Write the failing test**

Create `tests/test_parallel_client.py`:
```python
import json
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_parallel_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'services.parallel_client'"

**Step 3: Write minimal implementation**

Create `src/services/parallel_client.py`:
```python
from __future__ import annotations

from parallel import Parallel


class ParallelClient:
    """Thin wrapper around the Parallel AI Task API SDK."""

    def __init__(self, api_key: str):
        self.client = Parallel(api_key=api_key)

    def deep_research(
        self,
        prompt: str,
        processor: str = "pro",
        output_schema: dict | None = None,
    ) -> dict:
        """Submit a deep research task and block until results are ready.

        Args:
            prompt: The research query (max 15,000 chars).
            processor: Parallel AI processor tier ('pro' or 'ultra').
            output_schema: JSON Schema dict for structured output.
                           If None, Parallel AI auto-determines structure.

        Returns:
            The structured output dict from Parallel AI.
        """
        task_spec: dict | None = None
        if output_schema is not None:
            task_spec = {
                "output_schema": {
                    "type": "json",
                    "json_schema": output_schema,
                }
            }

        task_run = self.client.task_run.create(
            input=prompt,
            processor=processor,
            task_spec=task_spec,
        )

        result = self.client.task_run.result(task_run.run_id, api_timeout=3600)
        return result.output
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_parallel_client.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/services/parallel_client.py tests/test_parallel_client.py
git commit -m "feat: add ParallelClient wrapper for Parallel AI Task API"
```

---

### Task 3: Create CompetitorFetcherService — helpers and pure functions

Start with the pure logic: brief rendering, prompt building, result formatting. These have no external dependencies and are easy to test.

**Files:**
- Create: `src/services/competitor_fetcher.py`
- Create: `tests/test_competitor_fetcher.py`

**Step 1: Write the failing tests**

Create `tests/test_competitor_fetcher.py`:
```python
import json
from services.competitor_fetcher import (
    render_competitor_briefs,
    build_research_prompt,
    format_research_results_md,
    render_competitor_description,
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_competitor_fetcher.py -v`
Expected: FAIL with import error

**Step 3: Implement the helpers and service skeleton**

Create `src/services/competitor_fetcher.py`:
```python
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

from services.parallel_client import ParallelClient
from services.supabase_client import SupabaseService
from utils import load_env_value


# ---------------------------------------------------------------------------
# Pure helper functions (no external dependencies)
# ---------------------------------------------------------------------------

def render_competitor_briefs(competitors: list[dict]) -> str:
    """Convert Phase 1 competitor list into markdown text for Phase 2 prompt.

    Each competitor becomes a section with name, aliases, relevance,
    starting URLs, and campaign-specific focus areas.
    """
    if not competitors:
        return ""

    sections: list[str] = []
    for comp in competitors:
        name = comp["name"]
        aliases = comp.get("aliases", [])
        rationale = comp.get("rationale", "")
        urls = comp.get("discovered_urls", [])
        focus = comp.get("focus_areas", "")

        header = f"## {name}"
        if aliases:
            header += f" (also known as: {', '.join(aliases)})"

        lines = [header]
        if rationale:
            lines.append(f"**Why relevant:** {rationale}")
        if urls:
            lines.append("**Starting URLs:**")
            for url in urls:
                lines.append(f"- {url}")
        if focus:
            lines.append(f"**Focus areas:** {focus}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_research_prompt(
    template: str,
    campaign_info: str,
    custom_messages: str,
    competitor_briefs: str,
) -> str:
    """Inject variables into the research prompt template."""
    result = template
    result = result.replace("{{CAMPAIGN_INFO}}", campaign_info)
    result = result.replace("{{CUSTOM_MESSAGES}}", custom_messages)
    result = result.replace("{{COMPETITOR_BRIEFS}}", competitor_briefs)
    return result


def build_discovery_prompt(
    template: str,
    campaign_info: str,
    custom_messages: str,
    source_urls: str,
) -> str:
    """Inject variables into the discovery prompt template."""
    result = template
    result = result.replace("{{CAMPAIGN_INFO}}", campaign_info)
    result = result.replace("{{CUSTOM_MESSAGES}}", custom_messages)
    result = result.replace("{{SOURCE_URLS}}", source_urls)
    return result


def load_prompt_and_schema(prompt_dir: Path) -> tuple[str, dict]:
    """Load prompt text and JSON schema from a local prompt version directory.

    Returns the highest version found: (prompt_text, json_schema_dict).
    """
    prompt_files = sorted(prompt_dir.glob("v*_prompt.txt"), reverse=True)
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files found in {prompt_dir}")

    prompt_path = prompt_files[0]
    version = prompt_path.stem.split("_")[0]  # "v1"
    config_path = prompt_dir / f"{version}_config.json"

    prompt_text = prompt_path.read_text(encoding="utf-8")

    schema = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        schema = config.get("json_schema", {})

    return prompt_text, schema


def format_research_results_md(research_output: dict) -> str:
    """Format the structured JSON research output as readable markdown for display."""
    lines: list[str] = []

    overview = research_output.get("market_overview", "")
    if overview:
        lines.append(f"## Market Overview\n{overview}\n")

    for comp in research_output.get("competitors", []):
        name = comp.get("name", "Unknown")
        lines.append(f"### {name}")

        overview_text = comp.get("overview", "")
        if overview_text:
            lines.append(f"**Overview:** {overview_text}\n")

        packages = comp.get("pricing_packages", [])
        if packages:
            lines.append("**Pricing & Packages:**")
            for pkg in packages:
                lines.append(f"- **{pkg.get('name', '?')}**: {pkg.get('price', '?')}")
                lines.append(f"  {pkg.get('details', '')}")
                comparison = pkg.get("comparison_to_us", "")
                if comparison:
                    lines.append(f"  *Comparison:* {comparison}")
            lines.append("")

        pain_points = comp.get("pain_points", [])
        if pain_points:
            lines.append(f"**Pain Points ({len(pain_points)}):**")
            for pp in pain_points:
                lines.append(f"- **Claim:** {pp.get('claim', '?')}")
                lines.append(f"  **Story:** {pp.get('story', '?')}")
            lines.append("")

        news = comp.get("critical_news", [])
        if news:
            lines.append(f"**Critical News ({len(news)}):**")
            for item in news:
                lines.append(f"- [{item.get('date', '?')}] {item.get('headline', '?')}")
                lines.append(f"  {item.get('summary', '')}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


def render_competitor_description(research_data: dict) -> str:
    """Render a single competitor's research_data as readable text.

    Used by the state patcher's lookup_competitor tool.
    """
    lines: list[str] = []

    overview = research_data.get("overview", "")
    if overview:
        lines.append(f"**Overview:** {overview}\n")

    packages = research_data.get("pricing_packages", [])
    if packages:
        lines.append("**Pricing & Packages:**")
        for pkg in packages:
            lines.append(f"- {pkg.get('name', '?')}: {pkg.get('price', '?')} — {pkg.get('details', '')}")
            comparison = pkg.get("comparison_to_us", "")
            if comparison:
                lines.append(f"  Comparison: {comparison}")
        lines.append("")

    pain_points = research_data.get("pain_points", [])
    if pain_points:
        lines.append("**Known Issues:**")
        for pp in pain_points:
            lines.append(f"- {pp.get('claim', '?')}: {pp.get('story', '?')}")
        lines.append("")

    news = research_data.get("critical_news", [])
    if news:
        lines.append("**Recent News:**")
        for item in news:
            lines.append(f"- [{item.get('date', '?')}] {item.get('headline', '?')}: {item.get('summary', '')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini grounding (Phase 1)
# ---------------------------------------------------------------------------

def _discover_competitors_gemini(
    api_key: str,
    prompt: str,
    model: str = "gemini-2.5-pro",
) -> list[dict]:
    """Call Gemini with Google Search grounding to discover competitors.

    Returns list of competitor dicts from the JSON response.
    """
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
        ),
    )

    text = response.text or ""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)

    competitors = json.loads(text.strip())
    if not isinstance(competitors, list):
        raise ValueError(f"Expected JSON array from discovery, got {type(competitors)}")

    return competitors


# ---------------------------------------------------------------------------
# Service class — the main public API
# ---------------------------------------------------------------------------

class CompetitorFetcherService:
    """Generates structured competitor intelligence for a campaign.

    Usage (programmatic):
        service = CompetitorFetcherService(
            gemini_api_key="...",
            parallel_api_key="...",
            supabase_service=supabase_svc,
        )
        result = service.generate_competitor_knowledge(
            campaign_id="elg_b2c",
            campaign_info="<campaign info text from Langfuse>",
        )

    Usage (CLI): see scripts/run_competitor_fetcher.py
    """

    def __init__(
        self,
        gemini_api_key: str,
        parallel_api_key: str,
        supabase_service: Optional[SupabaseService] = None,
        prompts_dir: Optional[Path] = None,
    ):
        self.gemini_api_key = gemini_api_key
        self.parallel_client = ParallelClient(api_key=parallel_api_key)
        self.supabase_service = supabase_service
        self.prompts_dir = prompts_dir or Path(__file__).resolve().parent.parent.parent / "prompts"

    def discover_competitors(
        self,
        campaign_info: str,
        custom_messages: str = "",
        source_urls: str = "",
        model: str = "gemini-2.5-pro",
    ) -> list[dict]:
        """Phase 1: Discover competitors using Gemini + Google Search grounding.

        Returns list of competitor dicts with:
            name, aliases, rationale, discovered_urls, focus_areas
        """
        discovery_dir = self.prompts_dir / "competitor_discovery"
        template, _ = load_prompt_and_schema(discovery_dir)

        # Format optional blocks
        source_block = ""
        if source_urls:
            source_block = "# Reference URLs\nUse these as additional research sources:\n"
            for url in source_urls.strip().splitlines():
                source_block += f"- {url}\n"

        msg_block = ""
        if custom_messages:
            msg_block = "# Additional Market Knowledge\n" + custom_messages

        prompt = build_discovery_prompt(
            template=template,
            campaign_info=campaign_info,
            custom_messages=msg_block,
            source_urls=source_block,
        )

        return _discover_competitors_gemini(
            api_key=self.gemini_api_key,
            prompt=prompt,
            model=model,
        )

    def research_competitors(
        self,
        campaign_info: str,
        competitors: list[dict],
        custom_messages: str = "",
        processor: str = "pro",
    ) -> dict:
        """Phase 2: Deep research via Parallel AI Task API.

        Args:
            campaign_info: Campaign context text.
            competitors: Phase 1 output (list of competitor dicts with URLs/focus).
            custom_messages: Extra operator context.
            processor: Parallel AI tier ('pro' or 'ultra').

        Returns:
            Structured dict: {market_overview: str, competitors: [...]}
        """
        research_dir = self.prompts_dir / "competitor_deep_research"
        template, schema = load_prompt_and_schema(research_dir)

        msg_block = ""
        if custom_messages:
            msg_block = "# Additional Market Knowledge\n" + custom_messages

        competitor_briefs = render_competitor_briefs(competitors)

        prompt = build_research_prompt(
            template=template,
            campaign_info=campaign_info,
            custom_messages=msg_block,
            competitor_briefs=competitor_briefs,
        )

        result = self.parallel_client.deep_research(
            prompt=prompt,
            processor=processor,
            output_schema=schema if schema else None,
        )

        # Handle string output (parse JSON if needed)
        if isinstance(result, str):
            result = json.loads(result)

        return result

    def store_results(
        self,
        campaign_id: str,
        research_output: dict,
        phase1_competitors: list[dict],
    ) -> int:
        """Phase 3: Store per-competitor data in Supabase.

        Args:
            campaign_id: Campaign to store results for.
            research_output: Phase 2 output dict.
            phase1_competitors: Phase 1 competitor list (for alias fallback).

        Returns:
            Number of competitors stored.

        Raises:
            ValueError: If no supabase_service was provided.
        """
        if self.supabase_service is None:
            raise ValueError("Cannot store results: no SupabaseService provided.")

        rows = []
        for comp in research_output.get("competitors", []):
            # Find matching Phase 1 competitor for alias fallback
            phase1_match = next(
                (c for c in phase1_competitors if c["name"].lower() == comp.get("name", "").lower()),
                None,
            )
            aliases = comp.get("aliases", [])
            if not aliases and phase1_match:
                aliases = phase1_match.get("aliases", [])

            rows.append({
                "competitor_id": comp.get("competitor_id", comp.get("name", "unknown").lower().replace(" ", "_")),
                "name": comp.get("name", "Unknown"),
                "aliases": aliases,
                "research_data": {
                    "overview": comp.get("overview", ""),
                    "pricing_packages": comp.get("pricing_packages", []),
                    "pain_points": comp.get("pain_points", []),
                    "critical_news": comp.get("critical_news", []),
                },
            })

        return self.supabase_service.upsert_competitors(campaign_id, rows)

    def generate_competitor_knowledge(
        self,
        campaign_id: str,
        campaign_info: str,
        custom_messages: str = "",
        source_urls: str = "",
        processor: str = "pro",
        competitors_override: list[dict] | None = None,
        store: bool = True,
    ) -> dict:
        """Full pipeline: discover competitors, research them, store results.

        This is the main entry point for programmatic use.

        Args:
            campaign_id: Campaign identifier.
            campaign_info: Campaign context text (typically from Langfuse).
            custom_messages: Extra operator-provided market knowledge.
            source_urls: Newline-separated reference URLs.
            processor: Parallel AI tier ('pro' or 'ultra').
            competitors_override: Skip Phase 1 and use these competitors directly.
                Useful when you already know the competitor list.
            store: Whether to store results in Supabase (requires supabase_service).

        Returns:
            The structured research output:
            {
                "market_overview": "...",
                "competitors": [
                    {
                        "name": "Norlys",
                        "competitor_id": "norlys",
                        "aliases": [...],
                        "overview": "...",
                        "pricing_packages": [...],
                        "pain_points": [...],
                        "critical_news": [...]
                    },
                    ...
                ]
            }
        """
        # Phase 1: Discovery
        if competitors_override is not None:
            competitors = competitors_override
        else:
            competitors = self.discover_competitors(
                campaign_info=campaign_info,
                custom_messages=custom_messages,
                source_urls=source_urls,
            )

        # Phase 2: Deep research
        research_output = self.research_competitors(
            campaign_info=campaign_info,
            competitors=competitors,
            custom_messages=custom_messages,
            processor=processor,
        )

        # Phase 3: Store
        if store and self.supabase_service is not None:
            self.store_results(campaign_id, research_output, competitors)

        return research_output
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_competitor_fetcher.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add src/services/competitor_fetcher.py tests/test_competitor_fetcher.py
git commit -m "feat: add CompetitorFetcherService with generate_competitor_knowledge API"
```

---

### Task 4: Add test for Gemini discovery (mocked)

**Files:**
- Modify: `tests/test_competitor_fetcher.py`

**Step 1: Write the test**

Add to `tests/test_competitor_fetcher.py`:
```python
from unittest.mock import MagicMock, patch
from services.competitor_fetcher import _discover_competitors_gemini


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
```

**Step 2: Run tests**

Run: `pytest tests/test_competitor_fetcher.py -v`
Expected: 9 passed

**Step 3: Commit**

```bash
git add tests/test_competitor_fetcher.py
git commit -m "test: add Gemini discovery mock tests"
```

---

### Task 5: Update Supabase client for research_data schema

**Files:**
- Modify: `src/services/supabase_client.py`
- Create: `tests/test_supabase_competitors.py`

**Step 1: Write the failing tests**

Create `tests/test_supabase_competitors.py`:
```python
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

    # Set up chainable mock
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
    # Verify table was called for both delete and insert
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_supabase_competitors.py -v`
Expected: FAIL with AttributeError

**Step 3: Add methods to SupabaseService**

Add to `src/services/supabase_client.py` (inside the `SupabaseService` class, after existing methods):

```python
    def upsert_competitors(self, campaign_id: str, competitors: list[dict]) -> int:
        """Replace all competitors for a campaign with fresh data.

        Deletes existing rows then inserts new ones.

        Args:
            campaign_id: The campaign to update.
            competitors: List of dicts with: competitor_id, name, aliases, research_data.

        Returns:
            Number of competitors inserted.
        """
        # Delete existing
        try:
            self.client.table("campaign_competitors")\
                .delete()\
                .eq("campaign_id", campaign_id)\
                .execute()
        except Exception as e:
            logger.warning(f"Error deleting old competitors for {campaign_id}: {e}")

        # Insert new
        rows = []
        for comp in competitors:
            rows.append({
                "campaign_id": campaign_id,
                "competitor_id": comp["competitor_id"],
                "name": comp["name"],
                "aliases": comp.get("aliases", []),
                "research_data": comp.get("research_data", {}),
            })

        if rows:
            try:
                self.client.table("campaign_competitors").insert(rows).execute()
            except Exception as e:
                logger.error(f"Error inserting competitors for {campaign_id}: {e}")
                raise

        return len(rows)

    def get_competitor_research_data(self, campaign_id: str, competitor_id: str) -> Optional[dict]:
        """Get the structured research_data for a specific competitor."""
        try:
            response = self.client.table("campaign_competitors")\
                .select("research_data")\
                .eq("campaign_id", campaign_id)\
                .eq("competitor_id", competitor_id)\
                .single()\
                .execute()
            if response.data:
                return response.data.get("research_data")
            return None
        except Exception as e:
            if "PGRST116" not in str(e):
                logger.warning(f"Error fetching research data for {competitor_id}: {e}")
            return None
```

**Step 4: Update existing get_competitor_description for backward compat**

Replace the existing `get_competitor_description` method in `src/services/supabase_client.py`:

```python
    def get_competitor_description(self, campaign_id: str, competitor_id: str) -> Optional[str]:
        """Get a human-readable description for a competitor.

        Reads from research_data (jsonb) and renders to text.
        Falls back to description_md if research_data is not available.
        """
        try:
            response = self.client.table("campaign_competitors")\
                .select("research_data, description_md")\
                .eq("campaign_id", campaign_id)\
                .eq("competitor_id", competitor_id)\
                .single()\
                .execute()
            if response.data:
                research_data = response.data.get("research_data")
                if research_data:
                    from services.competitor_fetcher import render_competitor_description
                    return render_competitor_description(research_data)
                return response.data.get("description_md")
            return None
        except Exception as e:
            if "PGRST116" not in str(e):
                logger.warning(f"Error fetching competitor description for {competitor_id}: {e}")
            return None
```

**Step 5: Run tests**

Run: `pytest tests/test_supabase_competitors.py -v`
Expected: 2 passed

**Step 6: Commit**

```bash
git add src/services/supabase_client.py tests/test_supabase_competitors.py
git commit -m "feat: add structured competitor storage (research_data) to Supabase client"
```

---

### Task 6: Create the CLI script (thin wrapper)

The CLI collects interactive input and calls `CompetitorFetcherService` methods. All business logic lives in the service.

**Files:**
- Create: `scripts/run_competitor_fetcher.py`

**Step 1: Create the CLI script**

Create `scripts/run_competitor_fetcher.py`:
```python
#!/usr/bin/env python3
"""
Interactive CLI for generating competitor intelligence.

Thin wrapper around CompetitorFetcherService — collects user input,
displays results, and handles the human checkpoint between phases.

Usage:
    cd scripts
    python run_competitor_fetcher.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
os.chdir(_PROJECT_ROOT)

from services.competitor_fetcher import (
    CompetitorFetcherService,
    format_research_results_md,
)
from services.langfuse import init_langfuse, _with_rate_limit_backoff
from services.supabase_client import SupabaseService
from utils import load_env_value


# ---------------------------------------------------------------------------
# ANSI color helpers (same pattern as run_state_patcher.py)
# ---------------------------------------------------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def info(msg): print(f"{C.CYAN}[info]{C.RESET} {msg}")
def success(msg): print(f"{C.GREEN}[ok]{C.RESET} {msg}")
def warn(msg): print(f"{C.YELLOW}[warn]{C.RESET} {msg}")
def error(msg): print(f"{C.RED}[error]{C.RESET} {msg}")


def header(msg):
    print(f"\n{C.BOLD}{C.MAGENTA}{'=' * 50}")
    print(f"  {msg}")
    print(f"{'=' * 50}{C.RESET}\n")


def prompt_string(question, default=""):
    suffix = f" [{default}]" if default else ""
    raw = input(f"{C.BOLD}{question}{suffix}: {C.RESET}").strip()
    return raw if raw else default


def prompt_confirm(question, default=True):
    hint = "Y/n" if default else "y/N"
    raw = input(f"{C.BOLD}{question} ({hint}): {C.RESET}").strip().lower()
    return raw in ("y", "yes") if raw else default


def collect_multiline(label):
    print(f"{C.BOLD}{label} (empty line to finish):{C.RESET}")
    lines = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_competitors(competitors):
    """Print the Phase 1 competitor list with details."""
    for i, comp in enumerate(competitors, 1):
        name = comp.get("name", "?")
        aliases = comp.get("aliases", [])
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        urls = comp.get("discovered_urls", [])
        focus = comp.get("focus_areas", "")
        print(f"  {C.CYAN}{i}{C.RESET}) {C.BOLD}{name}{C.RESET}{alias_str}")
        if urls:
            short_urls = [u.split("//")[1] if "//" in u else u for u in urls[:3]]
            print(f"     URLs: {', '.join(short_urls)}")
        if focus:
            print(f"     Focus: {C.DIM}{focus[:120]}{'...' if len(focus) > 120 else ''}{C.RESET}")
        print()


def edit_competitor_list(competitors):
    """Interactive editing of the competitor list. Returns modified list."""
    if not prompt_confirm("Edit competitor list?", default=False):
        return competitors

    remove_input = prompt_string("Remove (comma-separated numbers, or empty)")
    if remove_input:
        indices = {int(x.strip()) - 1 for x in remove_input.split(",") if x.strip().isdigit()}
        competitors = [c for i, c in enumerate(competitors) if i not in indices]
        info(f"Removed {len(indices)} competitor(s). {len(competitors)} remaining.")

    while True:
        add_name = prompt_string("Add competitor name (empty to finish)")
        if not add_name:
            break
        competitors.append({
            "name": add_name, "aliases": [], "rationale": "Manually added",
            "discovered_urls": [], "focus_areas": "",
        })
        info(f"Added '{add_name}'")

    return competitors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    header("Competitor Intelligence Fetcher")

    # --- Gather inputs ---
    campaign_id = prompt_string("Campaign ID")
    if not campaign_id:
        error("Campaign ID is required.")
        sys.exit(1)

    prompt_name = prompt_string("Langfuse prompt name", default=f"{campaign_id}_info")

    # Fetch campaign info from Langfuse
    info("Initializing Langfuse...")
    langfuse_client = init_langfuse(push_to_langfuse=False)
    if not langfuse_client:
        error("Could not initialize Langfuse.")
        sys.exit(1)

    info(f"Fetching '{prompt_name}'...")
    try:
        prompt_obj = _with_rate_limit_backoff(
            lambda: langfuse_client.get_prompt(name=prompt_name, type="text")
        )
        campaign_info = prompt_obj.prompt
        success(f"Fetched campaign info ({len(campaign_info)} chars)")
    except Exception as exc:
        error(f"Failed to fetch prompt: {exc}")
        sys.exit(1)

    custom_messages = collect_multiline("Custom messages (optional)")
    source_urls = collect_multiline("Source URLs (optional)")

    # --- Initialize service ---
    gemini_key = load_env_value("GEMINI_API_KEY")
    parallel_key = load_env_value("PARALLEL_API_KEY")
    if not gemini_key:
        error("GEMINI_API_KEY is required.")
        sys.exit(1)
    if not parallel_key:
        error("PARALLEL_API_KEY is required.")
        sys.exit(1)

    sb_url = load_env_value("SUPABASE_URL")
    sb_key = load_env_value("SUPABASE_SERVICE_ROLE_KEY")
    supabase_service = SupabaseService(sb_url, sb_key) if sb_url and sb_key else None

    service = CompetitorFetcherService(
        gemini_api_key=gemini_key,
        parallel_api_key=parallel_key,
        supabase_service=supabase_service,
    )

    # --- Phase 1 ---
    header("Phase 1: Discovering Competitors (Gemini + Search)")
    t0 = time.perf_counter()
    try:
        competitors = service.discover_competitors(
            campaign_info=campaign_info,
            custom_messages=custom_messages,
            source_urls=source_urls,
        )
    except Exception as exc:
        error(f"Discovery failed: {exc}")
        sys.exit(1)
    success(f"Found {len(competitors)} competitors ({time.perf_counter() - t0:.1f}s)")

    print()
    display_competitors(competitors)
    competitors = edit_competitor_list(competitors)

    if not competitors:
        error("No competitors to research.")
        sys.exit(1)

    # --- Phase 2 ---
    print(f"\n{C.BOLD}Processor tier:{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}) pro  — $0.10, 5-15 min (default)")
    print(f"  {C.CYAN}2{C.RESET}) ultra — $0.30, 10-25 min")
    processor = "ultra" if prompt_string("Select", default="1") == "2" else "pro"

    header(f"Phase 2: Deep Research (Parallel AI, {processor})")
    info("Submitting research task (this may take 5-25 minutes)...")
    t0 = time.perf_counter()
    try:
        research_output = service.research_competitors(
            campaign_info=campaign_info,
            competitors=competitors,
            custom_messages=custom_messages,
            processor=processor,
        )
    except Exception as exc:
        error(f"Research failed: {exc}")
        sys.exit(1)
    elapsed = time.perf_counter() - t0
    success(f"Research complete ({elapsed / 60:.1f} min)")

    # --- Display results ---
    header("Results")
    print(format_research_results_md(research_output))
    info(f"{len(research_output.get('competitors', []))} competitors researched")

    # --- Phase 3 ---
    if supabase_service and prompt_confirm("Save to Supabase?"):
        count = service.store_results(campaign_id, research_output, competitors)
        success(f"Stored {count} competitors for '{campaign_id}'")
    else:
        local_path = _PROJECT_ROOT / "customer_data" / campaign_id / "competitor_research.json"
        if prompt_confirm(f"Save locally to {local_path}?", default=False):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(research_output, f, ensure_ascii=False, indent=2)
            success(f"Saved to {local_path}")

    header("Done!")


if __name__ == "__main__":
    main()
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/run_competitor_fetcher.py').read()); print('OK')"`
Expected: "OK"

**Step 3: Commit**

```bash
git add scripts/run_competitor_fetcher.py
git commit -m "feat: add competitor fetcher CLI (thin wrapper around service)"
```

---

### Task 7: Supabase schema migration

Add `research_data` column to `campaign_competitors` table.

**Step 1: Run migration SQL in Supabase dashboard**

```sql
ALTER TABLE campaign_competitors
ADD COLUMN IF NOT EXISTS research_data jsonb DEFAULT '{}';
```

**Step 2: Verify**

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'campaign_competitors'
ORDER BY ordinal_position;
```

---

### Task 8: End-to-end test

**Step 1: Verify env vars**

Run: `cd /Users/mikkeldahl/callbuddy_service && python -c "from utils import load_env_value; print('GEMINI:', bool(load_env_value('GEMINI_API_KEY'))); print('PARALLEL:', bool(load_env_value('PARALLEL_API_KEY'))); print('SUPABASE:', bool(load_env_value('SUPABASE_URL')))"`

**Step 2: Run the CLI**

Run: `cd scripts && python run_competitor_fetcher.py`

Test with:
- Campaign ID: `elg_b2c`
- Custom messages: `Alle danske elselskaber koeber fra Nordpool`
- Processor: `pro`

**Step 3: Verify Supabase**

```sql
SELECT competitor_id, name, aliases,
       jsonb_array_length(research_data->'pricing_packages') as packages,
       jsonb_array_length(research_data->'pain_points') as pain_points
FROM campaign_competitors WHERE campaign_id = 'elg_b2c';
```

**Step 4: Verify state patcher compatibility**

Run `run_state_patcher.py` with campaign_id `elg_b2c` and confirm `lookup_competitor` works.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: competitor intelligence fetcher — complete pipeline"
```
