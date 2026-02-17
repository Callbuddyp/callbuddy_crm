from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

from services.parallel_client import ParallelClient
from services.supabase_client import SupabaseService


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


def build_evaluator_prompt(
    template: str,
    campaign_info: str,
    researcher_output: str,
) -> str:
    """Inject variables into the evaluator prompt template."""
    result = template
    result = result.replace("{{CAMPAIGN_INFO}}", campaign_info)
    result = result.replace("{{RESEARCHER_OUTPUT}}", researcher_output)
    return result


def normalize_competitor(comp: dict) -> dict:
    """Normalize a competitor dict to flat v2 schema.

    Gemini with Google Search grounding ignores response_schema, so the
    model sometimes returns nested structures. This function flattens
    known nested patterns deterministically.
    """
    out: dict = {}

    # --- Identification: may be nested under "identification" key ---
    ident = comp.get("identification", {})
    out["name"] = comp.get("name") or ident.get("name", "Unknown")
    out["competitor_id"] = (
        comp.get("competitor_id")
        or ident.get("competitor_id")
        or out["name"].lower().replace(" ", "_").replace(".", "")
    )
    out["aliases"] = comp.get("aliases") or ident.get("aliases", [])

    # --- Comparison summary: may be a dict with sub-keys ---
    cs = comp.get("comparison_summary", "")
    if isinstance(cs, dict):
        parts = [v for v in cs.values() if isinstance(v, str) and v.strip()]
        out["comparison_summary"] = " ".join(parts)
    else:
        out["comparison_summary"] = cs or ""

    # --- Strengths: should be list of strings already ---
    out["strengths"] = comp.get("strengths", [])

    # --- Angles: should be list of strings already ---
    out["angles"] = comp.get("angles", [])

    # --- Switching barriers: may be dict {contractual: [...], practical: [...], ...} ---
    sb = comp.get("switching_barriers", [])
    if isinstance(sb, dict):
        flat: list[str] = []
        for items in sb.values():
            if isinstance(items, list):
                flat.extend(items)
            elif isinstance(items, str):
                flat.append(items)
        out["switching_barriers"] = flat
    else:
        out["switching_barriers"] = sb

    # --- Anecdotes: should be list of strings already ---
    out["anecdotes"] = comp.get("anecdotes", [])

    # --- Recent news: may be list of dicts {date, news_item/headline/summary} ---
    rn = comp.get("recent_news", [])
    normalized_news: list[str] = []
    for item in rn:
        if isinstance(item, str):
            normalized_news.append(item)
        elif isinstance(item, dict):
            date = item.get("date", "")
            text = item.get("news_item") or item.get("headline") or item.get("summary", "")
            if date and text:
                normalized_news.append(f"[{date}] {text}")
            elif text:
                normalized_news.append(text)
    out["recent_news"] = normalized_news

    return out


def normalize_competitors(competitors: list[dict]) -> list[dict]:
    """Normalize a list of competitor dicts to flat v2 schema."""
    return [normalize_competitor(c) for c in competitors]


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


def format_research_results_md(research_output: dict | list) -> str:
    """Format the structured JSON research output as readable markdown for display.

    Handles both v1 format (dict with market_overview + competitors) and
    v2 format (list of competitor dicts, or dict with competitors list of v2 dicts).
    """
    lines: list[str] = []

    # Handle v2 format (list of competitors directly)
    if isinstance(research_output, list):
        for comp in research_output:
            lines.append(render_competitor_description(comp))
            lines.append("\n---\n")
        return "\n".join(lines)

    overview = research_output.get("market_overview", "")
    if overview:
        lines.append(f"## Market Overview\n{overview}\n")

    competitors = research_output.get("competitors", [])

    # Detect v2 competitor format
    if competitors and "angles" in competitors[0]:
        for comp in competitors:
            lines.append(render_competitor_description(comp))
            lines.append("\n---\n")
        return "\n".join(lines)

    # v1 format
    for comp in competitors:
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
    """Render competitor data as readable text for the state patcher's lookup_competitor tool.

    Supports both v2 format (comparison_summary, angles, strengths, etc.)
    and v1 format (overview, pricing_packages, pain_points, critical_news).
    Detects format by checking for the 'angles' key (v2) vs 'pricing_packages' key (v1).
    """
    if "angles" in research_data or "comparison_summary" in research_data:
        return _render_v2(research_data)
    return _render_v1(research_data)


def _render_v2(data: dict) -> str:
    """Render v2 competitor data (sales-oriented format)."""
    lines: list[str] = []

    name = data.get("name", "")
    aliases = data.get("aliases", [])
    if name:
        header = f"**{name}**"
        if aliases:
            header += f" (også kendt som: {', '.join(aliases)})"
        lines.append(header)
        lines.append("")

    summary = data.get("comparison_summary", "")
    if summary:
        lines.append(f"**Sammenligning:** {summary}")
        lines.append("")

    strengths = data.get("strengths", [])
    if strengths:
        lines.append("**De er gode til:**")
        for s in strengths:
            lines.append(f"- {s}")
        lines.append("")

    angles = data.get("angles", [])
    if angles:
        lines.append("**Vores fordele:**")
        for a in angles:
            lines.append(f"- {a}")
        lines.append("")

    barriers = data.get("switching_barriers", [])
    if barriers:
        lines.append("**Skiftbarrierer:**")
        for b in barriers:
            lines.append(f"- {b}")
        lines.append("")

    anecdotes = data.get("anecdotes", [])
    if anecdotes:
        lines.append("**Historier:**")
        for a in anecdotes:
            lines.append(f"- {a}")
        lines.append("")

    news = data.get("recent_news", [])
    if news:
        lines.append("**Seneste nyt:**")
        for n in news:
            lines.append(f"- {n}")

    return "\n".join(lines)


def _render_v1(data: dict) -> str:
    """Render v1 competitor data (factsheet format, backwards compat)."""
    lines: list[str] = []

    overview = data.get("overview", "")
    if overview:
        lines.append(f"**Overview:** {overview}\n")

    packages = data.get("pricing_packages", [])
    if packages:
        lines.append("**Pricing & Packages:**")
        for pkg in packages:
            lines.append(f"- {pkg.get('name', '?')}: {pkg.get('price', '?')} — {pkg.get('details', '')}")
            comparison = pkg.get("comparison_to_us", "")
            if comparison:
                lines.append(f"  Comparison: {comparison}")
        lines.append("")

    pain_points = data.get("pain_points", [])
    if pain_points:
        lines.append("**Known Issues:**")
        for pp in pain_points:
            lines.append(f"- {pp.get('claim', '?')}: {pp.get('story', '?')}")
        lines.append("")

    news = data.get("critical_news", [])
    if news:
        lines.append("**Recent News:**")
        for item in news:
            lines.append(f"- [{item.get('date', '?')}] {item.get('headline', '?')}: {item.get('summary', '')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini grounding (used by both discovery and v2 researcher/evaluator)
# ---------------------------------------------------------------------------

def _call_gemini_with_grounding(
    api_key: str,
    prompt: str,
    model: str = "gemini-2.5-pro",
) -> list[dict]:
    """Call Gemini with Google Search grounding and parse JSON array response.

    Used by both the researcher and evaluator phases.
    Returns list of dicts from the JSON response.
    """
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    text = response.text or ""
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)

    try:
        competitors = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse Gemini response as JSON: {e}\n"
            f"Raw text (first 200 chars): {text[:200]}"
        ) from e
    if not isinstance(competitors, list):
        raise ValueError(f"Expected JSON array, got {type(competitors)}")

    return normalize_competitors(competitors)


# Keep old name as alias for backwards compatibility with existing tests
_discover_competitors_gemini = _call_gemini_with_grounding


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
        parallel_api_key: str = "",
        supabase_service: Optional[SupabaseService] = None,
        prompts_dir: Optional[Path] = None,
    ):
        self.gemini_api_key = gemini_api_key
        self.parallel_client = ParallelClient(api_key=parallel_api_key) if parallel_api_key else None
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

        return _call_gemini_with_grounding(
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
        if self.parallel_client is None:
            raise ValueError("Cannot run v1 research: no parallel_api_key provided.")

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
            store: Whether to store results in Supabase (requires supabase_service).

        Returns:
            The structured research output dict.
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

    # ------------------------------------------------------------------
    # v2 pipeline methods
    # ------------------------------------------------------------------

    def research_competitors_v2(
        self,
        campaign_info: str,
        custom_messages: str = "",
        source_urls: str = "",
        model: str = "gemini-2.5-pro",
    ) -> list[dict]:
        """Phase 1 (v2): Research competitors using Gemini + Google Search grounding.

        Returns list of competitor dicts with the v2 schema:
            name, competitor_id, aliases, comparison_summary, strengths,
            angles, switching_barriers, anecdotes, recent_news
        """
        researcher_dir = self.prompts_dir / "competitor_researcher"
        template, _ = load_prompt_and_schema(researcher_dir)

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

        return _call_gemini_with_grounding(
            api_key=self.gemini_api_key,
            prompt=prompt,
            model=model,
        )

    def evaluate_competitors(
        self,
        campaign_info: str,
        researcher_output: list[dict],
        model: str = "gemini-2.5-pro",
    ) -> list[dict]:
        """Phase 2 (v2): Evaluate and improve researcher output.

        Returns the corrected and improved list of competitor dicts.
        """
        evaluator_dir = self.prompts_dir / "competitor_evaluator"
        template, _ = load_prompt_and_schema(evaluator_dir)

        prompt = build_evaluator_prompt(
            template=template,
            campaign_info=campaign_info,
            researcher_output=json.dumps(researcher_output, ensure_ascii=False, indent=2),
        )

        return _call_gemini_with_grounding(
            api_key=self.gemini_api_key,
            prompt=prompt,
            model=model,
        )

    def store_results_v2(
        self,
        campaign_id: str,
        competitors: list[dict],
    ) -> int:
        """Store v2 competitor data in Supabase.

        Each competitor dict is stored directly as research_data.

        Returns:
            Number of competitors stored.
        """
        if self.supabase_service is None:
            raise ValueError("Cannot store results: no SupabaseService provided.")

        rows = []
        for comp in competitors:
            rows.append({
                "competitor_id": comp.get("competitor_id", comp.get("name", "unknown").lower().replace(" ", "_")),
                "name": comp.get("name", "Unknown"),
                "aliases": comp.get("aliases", []),
                "research_data": comp,
            })

        return self.supabase_service.upsert_competitors(campaign_id, rows)

    def generate_competitor_knowledge_v2(
        self,
        campaign_id: str,
        campaign_info: str,
        custom_messages: str = "",
        source_urls: str = "",
        model: str = "gemini-2.5-pro",
        skip_evaluator: bool = False,
        store: bool = True,
    ) -> list[dict]:
        """Full v2 pipeline: researcher -> evaluator -> store.

        Args:
            campaign_id: Campaign identifier.
            campaign_info: Campaign context text (from Langfuse or local file).
            custom_messages: Extra operator-provided market knowledge.
            source_urls: Newline-separated reference URLs.
            model: Gemini model to use for both phases.
            skip_evaluator: If True, skip the evaluator phase (for debugging).
            store: Whether to store results in Supabase.

        Returns:
            Final list of competitor dicts (v2 schema).
        """
        # Phase 1: Research
        competitors = self.research_competitors_v2(
            campaign_info=campaign_info,
            custom_messages=custom_messages,
            source_urls=source_urls,
            model=model,
        )

        # Phase 2: Evaluate
        if not skip_evaluator:
            competitors = self.evaluate_competitors(
                campaign_info=campaign_info,
                researcher_output=competitors,
                model=model,
            )

        # Phase 3: Store
        if store and self.supabase_service is not None:
            self.store_results_v2(campaign_id, competitors)

        return competitors
