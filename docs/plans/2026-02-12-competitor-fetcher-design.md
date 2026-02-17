# Competitor Intelligence Fetcher — Design

**Date:** 2026-02-12
**Status:** Design complete
**Location:** `scripts/run_competitor_fetcher.py`

---

## Overview

CLI script that auto-generates competitor intelligence for a CallBuddy campaign. Uses a two-phase approach: Gemini with Google Search grounding discovers competitors and scouts URLs, then Parallel AI Task API conducts deep research using those leads. Output is structured JSON stored in Supabase, consumed by the state patcher's `lookup_competitor` tool during live calls.

**Run after campaign creation. Re-run monthly for freshness.**

---

## Invocation

```bash
cd scripts
python run_competitor_fetcher.py
```

**Interactive CLI (same pattern as `run_state_patcher.py`):**

| Input | Source | Required |
|-------|--------|----------|
| Campaign ID | User input | Yes |
| Campaign info prompt name | Langfuse | Yes (default: `{campaign_id}_info`) |
| Custom messages | User input | Optional — extra market knowledge, directions |
| Source URLs | User input | Optional — e.g., elpris.dk, trustpilot.com |
| Processor tier | Menu selection | Yes (default: `pro`) |

---

## Pipeline

### Phase 1: Competitor Discovery (Gemini + Google Search Grounding)

Single Gemini call with web search grounding enabled.

**Input:**
- `{{CAMPAIGN_INFO}}` from Langfuse
- `{{CUSTOM_MESSAGES}}` from operator
- `{{SOURCE_URLS}}` from operator

**Task:** Identify relevant competitors in Denmark AND scout real URLs + focus areas for each.

**Output (JSON array):**
```json
[
  {
    "name": "Norlys",
    "aliases": ["Eniig", "Stofa"],
    "rationale": "Largest energy company, direct el competitor",
    "discovered_urls": [
      "https://norlys.dk/el/produkter",
      "https://dk.trustpilot.com/review/norlys.dk",
      "https://www.dr.dk/nyheder/norlys-prisstigninger-2025"
    ],
    "focus_areas": "Recently raised spot-tillaeg by 3 oere/kWh. Bundles gas+el with 6-month binding. Trustpilot 1.8 stars with recurring complaints about billing errors."
  }
]
```

**Human checkpoint:** Display discovered competitors. Operator can add/remove/edit before Phase 2.

**Cost:** ~$0.01 (single Gemini call with grounding)
**Latency:** ~5-10 seconds

**Why this matters for Phase 2:** The discovered URLs and focus areas are passed directly to Phase 2 as pre-research intelligence. This means Phase 2 doesn't waste tokens discovering what Phase 1 already found — it starts from verified URLs and campaign-specific angles, spending all its budget on depth instead of breadth.

---

### Phase 2: Deep Research (Parallel AI Task API)

Single Parallel AI Task API call researching ALL competitors together.

**Input:**
- Full research prompt with `{{CAMPAIGN_INFO}}`
- `{{CUSTOM_MESSAGES}}` incorporated
- `{{COMPETITOR_BRIEFS}}` — rendered from Phase 1 output, including discovered URLs and focus areas per competitor

**Parallel AI configuration:**
- Processor: `pro` ($0.10/request) or `ultra` ($0.30/request)
- Output schema: JSON (structured, not text/markdown)
- Timeout: up to 3600s (API polling)

**The research prompt instructs the agent to research 3 layers per competitor:**

1. **Pricing & Packages** — Starting from discovered pricing URLs. Direct apple-to-apple comparison with our product. Exact prices, binding, fees, contract terms.

2. **Pain Points & Stories** — Starting from discovered Trustpilot URLs. Verified negative reviews with specific anecdotes. Every pain point has a `claim` + `story` format. Generic complaints rejected.

3. **Critical News** — Starting from discovered news URLs. Forbrugerombudsmanden, price hikes, outages, scandals from last 12 months.

**Output:** Structured JSON matching the schema in `prompts/competitor_deep_research/v1_config.json`.

**Cost:** $0.10 (pro) or $0.30 (ultra) — single API call
**Latency:** 5-25 minutes depending on processor

---

### Phase 3: Storage

1. Parse Parallel AI JSON response directly into per-competitor objects
2. Display results for human review in CLI
3. On confirmation, upsert to Supabase:
   - Delete existing competitors for this campaign_id
   - Insert fresh rows from the structured JSON
4. No markdown parsing needed — JSON maps directly to table columns

---

## Prompts

Stored locally at `prompts/` and uploaded to Langfuse for versioning.

### competitor_discovery (Phase 1)
- **File:** `prompts/competitor_discovery/v1_prompt.txt`
- **Schema:** `prompts/competitor_discovery/v1_config.json`
- **Variables:** `CAMPAIGN_INFO`, `CUSTOM_MESSAGES`, `SOURCE_URLS`
- **Provider:** Gemini with Google Search grounding
- **Key design:** Asks for `discovered_urls` and `focus_areas` per competitor — these become the starting points for Phase 2

### competitor_deep_research (Phase 2)
- **File:** `prompts/competitor_deep_research/v1_prompt.txt`
- **Schema:** `prompts/competitor_deep_research/v1_config.json`
- **Variables:** `CAMPAIGN_INFO`, `CUSTOM_MESSAGES`, `COMPETITOR_BRIEFS`
- **Provider:** Parallel AI Task API
- **Key design:** Receives pre-research intelligence from Phase 1 via `COMPETITOR_BRIEFS`. Enforces claim+story format for pain points. Output is structured JSON, not markdown.

---

## Supabase Storage

### Table: `campaign_competitors`

| Column | Type | Description |
|--------|------|-------------|
| `id` | uuid | Auto-generated primary key |
| `campaign_id` | text | FK to campaigns |
| `competitor_id` | text | snake_case identifier (e.g., `norlys`) |
| `name` | text | Display name (e.g., `Norlys`) |
| `aliases` | jsonb | Alternative names: `["Eniig", "Stofa"]` |
| `research_data` | jsonb | Full structured data: overview, pricing_packages, pain_points, critical_news |
| `fetched_at` | timestamptz | Last fetch date |

**Unique constraint:** `(campaign_id, competitor_id)`

### research_data structure (jsonb)

```json
{
  "overview": "Largest energy company in Denmark...",
  "pricing_packages": [
    {
      "name": "Norlys Basis",
      "price": "49 kr/md + spot + 4.9 oere/kWh tillaeg",
      "details": "Ingen binding, 1 md opsigelse, 0 kr oprettelse",
      "comparison_to_us": "Their tillaeg is 4.9 oere vs our X oere — Y oere dyrere per kWh"
    }
  ],
  "pain_points": [
    {
      "claim": "Impossible to cancel — requires calling 3 separate times",
      "story": "A customer tried to cancel online but was told to call..."
    }
  ],
  "critical_news": [
    {
      "date": "2025-11",
      "headline": "Norlys haever spot-tillaeg med 3 oere",
      "summary": "Norlys raised their spot supplement affecting 200,000 customers..."
    }
  ]
}
```

### Upsert logic

On re-run (monthly refresh):
1. Delete all rows for this `campaign_id`
2. Insert fresh competitor rows from the new research
3. History is not preserved — each run is a full replacement

---

## lookup_competitor (State Patcher Tool)

When the state patcher needs competitor info during a live call:

```python
# Query by name or alias
result = supabase.table("campaign_competitors") \
    .select("name, aliases, research_data, fetched_at") \
    .eq("campaign_id", campaign_id) \
    .or_(f"name.ilike.%{query}%,aliases.cs.[\"{query}\"]") \
    .execute()
```

Returns `research_data` (structured JSON) so the state patcher can use specific fields:
- Full data for a comprehensive briefing
- Just `pricing_packages` for price comparison
- Just `pain_points` for objection handling ammunition

The `fetched_at` timestamp is included so the state patcher knows how fresh the data is.

---

## CLI Flow

```
$ python scripts/run_competitor_fetcher.py

=== Competitor Intelligence Fetcher ===

Campaign ID: elg_b2c
Langfuse prompt name [elg_b2c_info]:
  -> Fetched campaign info (2,340 chars)

Custom messages (optional, empty line to finish):
  > Alle danske elselskaber koeber fra Nordpool
  > Fokuser paa bindingsperioder
  >

Source URLs (optional, empty line to finish):
  > https://elpris.dk
  >

--- Phase 1: Discovering competitors (Gemini + Search) ---
Found 8 competitors:
  1. Norlys (aliases: Eniig, Stofa)
     URLs: norlys.dk/el/produkter, dk.trustpilot.com/review/norlys.dk
     Focus: Recently raised spot-tillaeg, 6-month binding on bundles

  2. Andel Energi (aliases: SEAS-NVE)
     URLs: andelenergi.dk/el, dk.trustpilot.com/review/andel.dk
     Focus: Eastern Denmark cooperative, competitive spot prices
  ...

Edit? [y/N]: y
Remove (comma-separated numbers, or empty):
Add competitor (empty to skip):
  >

Proceeding with 8 competitors.
Processor tier [pro/ultra] (pro): ultra

--- Phase 2: Deep Research (Parallel AI, ultra) ---
Task submitted. Polling for results...
[==============================] Complete (12m 34s)

--- Results ---
Market Overview: The Danish electricity market has 50+ providers...

### Norlys
Overview: Largest energy company in Denmark...
Pricing: Norlys Basis — 49 kr/md + spot + 4.9 oere/kWh
Pain Points: 3 found
News: 2 items

### Andel Energi
...

Save to Supabase? [Y/n]: y
  -> Stored 8 competitors for campaign elg_b2c
Done.
```

---

## File Structure

```
scripts/
    run_competitor_fetcher.py         # CLI entry point (interactive)

src/services/
    competitor_fetcher.py             # Orchestration: Phase 1 -> 2 -> 3
    parallel_client.py                # Parallel AI Task API wrapper

prompts/
    competitor_discovery/
        v1_prompt.txt                 # Phase 1 prompt
        v1_config.json                # Phase 1 JSON schema
    competitor_deep_research/
        v1_prompt.txt                 # Phase 2 prompt
        v1_config.json                # Phase 2 JSON schema
```

### parallel_client.py

Thin wrapper around the `parallel` Python SDK:

```python
from parallel import Parallel
from parallel.types import TaskSpecParam

class ParallelClient:
    def __init__(self, api_key: str):
        self.client = Parallel(api_key=api_key)

    def deep_research(self, prompt: str, processor: str = "pro", output_schema: dict = None) -> dict:
        """Submit deep research task and wait for structured JSON result."""
        task_spec = TaskSpecParam(output_schema=output_schema) if output_schema else None
        task_run = self.client.task_run.create(
            input=prompt,
            processor=processor,
            task_spec=task_spec
        )
        result = self.client.task_run.result(task_run.run_id, api_timeout=3600)
        return result.output
```

### competitor_fetcher.py

Main service orchestrating all three phases:

```python
class CompetitorFetcher:
    def __init__(self, parallel_client, gemini_client, supabase_client):
        ...

    def discover_competitors(self, campaign_info: str, custom_messages: str, source_urls: str) -> list[dict]:
        """Phase 1: Gemini + grounding -> competitor list with URLs and focus areas."""

    def render_competitor_briefs(self, competitors: list[dict]) -> str:
        """Convert Phase 1 output into COMPETITOR_BRIEFS text for Phase 2 prompt."""

    def research_competitors(self, campaign_info: str, competitor_briefs: str, custom_messages: str, processor: str, schema: dict) -> dict:
        """Phase 2: Parallel AI deep research -> structured JSON."""

    def store_results(self, campaign_id: str, research_output: dict, phase1_competitors: list[dict]):
        """Phase 3: Upsert per-competitor data to Supabase."""
```

---

## Cost Summary

| Phase | Service | Cost | Latency |
|-------|---------|------|---------|
| 1. Discovery | Gemini + grounding | ~$0.01 | 5-10s |
| 2. Deep research | Parallel AI (pro) | $0.10 | 5-15 min |
| 2. Deep research | Parallel AI (ultra) | $0.30 | 10-25 min |
| **Total (pro)** | | **~$0.11** | **~5-15 min** |
| **Total (ultra)** | | **~$0.31** | **~10-25 min** |

Monthly cost per campaign: $0.11-0.31

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes | Phase 1: Gemini with grounding |
| `PARALLEL_API_KEY` | Yes (new) | Phase 2: Parallel AI Task API |
| `SUPABASE_URL` | Yes | Phase 3: Storage |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Phase 3: Storage |
| `LANGFUSE_SECRET_KEY` | Yes | Fetching campaign info prompt |
| `LANGFUSE_PUBLIC_KEY` | Yes | Fetching campaign info prompt |
| `LANGFUSE_HOST` | Yes | Fetching campaign info prompt |

---

## Campaign Setup Workflow

```
1. Create campaign in Supabase (via onboard.py)
2. Create campaign info prompt in Langfuse (via onboard.py)
3. Run competitor fetcher:
   python scripts/run_competitor_fetcher.py
   -> Enter campaign ID
   -> Select Langfuse prompt name
   -> Add custom messages / URLs (optional)
   -> Review auto-discovered competitors (Phase 1)
   -> Confirm -> deep research (Phase 2)
   -> Review results -> store in Supabase (Phase 3)
4. Campaign ready — state patcher has lookup_competitor access
```

**Monthly refresh:** Same script, same campaign. Replaces previous data in Supabase.
