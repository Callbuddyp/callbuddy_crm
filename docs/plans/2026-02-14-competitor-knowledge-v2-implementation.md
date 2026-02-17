# Competitor Knowledge v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the two-phase competitor pipeline (discovery + Parallel AI deep research) with a researcher + evaluator chain using Gemini + Google Search grounding, producing sales-oriented competitor intelligence.

**Architecture:** Single service method calls Gemini twice — first a researcher that identifies competitors and produces structured intelligence with pre-computed comparative statements, then an evaluator that fills gaps and sharpens quality. Both use Google Search grounding. Output schema has 7 fields per competitor: comparison_summary, strengths, angles, switching_barriers, anecdotes, recent_news, plus identification fields.

**Tech Stack:** Python, Google Gemini API (genai SDK), Google Search grounding, Supabase (unchanged), pytest

---

### Task 1: Create the researcher prompt and schema

**Files:**
- Create: `prompts/competitor_researcher/v1_prompt.txt`
- Create: `prompts/competitor_researcher/v1_config.json`

**Step 1: Write the researcher prompt**

Create `prompts/competitor_researcher/v1_prompt.txt`:

```text
# Role

You are a competitive intelligence analyst preparing ammunition for sales agents. Your task is to research competitors relevant to the sales campaign below and produce actionable intelligence that helps a salesperson win deals during live phone calls.

This is NOT a market analysis report. Every piece of information you produce must answer one question: "How does this help our salesperson close a deal against this competitor?"

# Our Campaign

"""
{{CAMPAIGN_INFO}}
"""

{{CUSTOM_MESSAGES}}

{{SOURCE_URLS}}

# Task

Use web search to identify the competitors a customer might currently be using or might mention during a sales call. For each competitor, produce structured intelligence across these dimensions:

## 1. Identification
- **name**: Official company name as used in the market
- **competitor_id**: Stable snake_case identifier (e.g. "norlys", "andel_energi")
- **aliases**: Other names customers use — old brand names, abbreviations, sub-brands, common misspellings. A customer on the phone might say any of these.

## 2. Comparison Summary
A complete pricing and terms comparison against OUR offering. Cover ALL relevant pricing dimensions for this market — not just the headline price. Include subscription fees, per-unit costs, setup fees, binding periods, cancellation terms, discounts, and any other costs the customer would encounter. Reference our specific numbers from the Campaign Information above.

## 3. Strengths
What this competitor genuinely does well. The salesperson needs to know this so they can ACKNOWLEDGE it credibly rather than blindly attacking. A customer who loves their current provider's app or local presence will disengage if the salesperson dismisses it.

## 4. Angles
Pre-computed comparative statements the salesperson can use. Each angle must:
- Reference specific numbers (ours vs theirs)
- Be something a salesperson could naturally say during a call
- Highlight where WE are better, not just where THEY are worse

## 5. Switching Barriers
What holds customers with this competitor. Three types:
- **Contractual**: binding periods, cancellation fees, notice periods
- **Practical**: bundled services they'd lose, effort to switch
- **Psychological**: brand loyalty, "de er lokale", habit

The salesperson needs to anticipate and address these proactively.

## 6. Anecdotes
Concrete stories from customer reviews (Trustpilot, forums, social media) that create doubt about the competitor. Each anecdote must be:
- **Specific**: Names a situation, timeframe, and outcome — not "dårlig kundeservice"
- **Believable**: Paraphrased from real reviews, not fabricated
- **Usable**: Something a salesperson could reference: "Det hører vi faktisk tit fra folk der kommer fra [competitor]..."

Do NOT include generic complaints. A story about waiting 45 minutes on the phone after being told 11 minutes is good. "Mange klager over kundeservicen" is useless.

## 7. Recent News
Price changes, service issues, regulatory actions, or press from the last 12 months that the customer may not be aware of. Each item must be dated and specific.

# Guidelines
- Focus on competitors a customer would realistically mention on the phone
- Include both major players and relevant budget/niche alternatives
- Every URL you reference must be real — do not fabricate URLs
- All comparisons must use specific numbers from our Campaign Information
- If you cannot find reliable information for a field, use an empty array rather than guessing
- Write in the same language as the Campaign Information

# Output Format
Return ONLY a valid JSON array. No commentary, no markdown, just JSON.
```

**Step 2: Write the researcher config/schema**

Create `prompts/competitor_researcher/v1_config.json`:

```json
{
  "json_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "required": ["name", "competitor_id", "aliases", "comparison_summary", "strengths", "angles", "switching_barriers", "anecdotes", "recent_news"],
      "additionalProperties": false,
      "properties": {
        "name": {
          "type": "string",
          "description": "Official company name as used in the market"
        },
        "competitor_id": {
          "type": "string",
          "description": "Stable snake_case identifier, e.g. 'norlys', 'andel_energi'"
        },
        "aliases": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Other names customers use: old brands, abbreviations, sub-brands"
        },
        "comparison_summary": {
          "type": "string",
          "description": "Complete pricing and terms picture covering ALL relevant dimensions for this market, with direct comparison to our offering using specific numbers"
        },
        "strengths": {
          "type": "array",
          "items": { "type": "string" },
          "description": "What this competitor genuinely does well — things a customer might value and the salesperson should acknowledge"
        },
        "angles": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Pre-computed comparative statements referencing specific numbers from both our offering and the competitor's"
        },
        "switching_barriers": {
          "type": "array",
          "items": { "type": "string" },
          "description": "What holds customers with this competitor: contractual, practical, or psychological barriers"
        },
        "anecdotes": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Concrete stories from reviews/forums with specific details — not generic complaints"
        },
        "recent_news": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Dated news items from last 12 months: price changes, regulatory actions, service issues"
        }
      }
    }
  }
}
```

**Step 3: Commit**

```bash
git add prompts/competitor_researcher/
git commit -m "feat: add competitor researcher v1 prompt and schema"
```

---

### Task 2: Create the evaluator prompt and schema

**Files:**
- Create: `prompts/competitor_evaluator/v1_prompt.txt`
- Create: `prompts/competitor_evaluator/v1_config.json`

**Step 1: Write the evaluator prompt**

Create `prompts/competitor_evaluator/v1_prompt.txt`:

```text
# Role

You are a quality reviewer for competitive sales intelligence. A researcher has produced competitor data for a sales campaign. Your job is to evaluate, correct, and improve the output so every piece of information is accurate and actionable for a salesperson during a live phone call.

# Our Campaign

"""
{{CAMPAIGN_INFO}}
"""

# Researcher Output

"""
{{RESEARCHER_OUTPUT}}
"""

# Evaluation Checklist

Review each competitor against these criteria. Use web search to verify claims and fill gaps.

## 1. Comparison Summary Completeness
- Does it cover ALL relevant pricing dimensions for this market? (Not just headline price — check for subscription fees, per-unit costs, setup fees, binding, cancellation terms, included extras, discounts)
- Are the comparisons to our offering accurate? Cross-check against the Campaign Information above.
- If any pricing dimension is missing, search for it and add it.

## 2. Strengths Quality
- Are these genuine strengths a customer would actually value?
- Remove generic filler like "god kundeservice" unless backed by evidence.
- Add any notable strengths the researcher missed.

## 3. Angles Actionability
- Does each angle reference specific numbers from BOTH our offering and the competitor's?
- Could a salesperson naturally say this during a phone call?
- Remove angles that are too vague or don't help close a deal.
- Sharpen weak angles with concrete numbers.

## 4. Switching Barriers
- Are contractual barriers accurate? (Check binding periods, cancellation terms)
- Are psychological barriers realistic for this market?
- Add any barriers the researcher missed.

## 5. Anecdotes Quality
- Is each anecdote specific enough to use in conversation? (Has a situation, timeframe, outcome)
- Remove anything that reads as generic ("dårlig service", "lange ventetider" without details)
- If anecdotes are weak, search Trustpilot / forums for better ones.

## 6. Recent News
- Are dates and facts accurate?
- Search for any recent events the researcher missed (price changes, regulatory actions, press)
- Remove items older than 12 months.

## 7. Missing Competitors
- Are there major competitors in this market that the researcher missed entirely?
- If so, research and add them with the full field set.

## 8. Irrelevant Competitors
- Remove any competitors that a customer would realistically never mention.

# Output

Return the corrected and improved JSON array. Same schema as the input — every competitor must have all 9 fields. Maintain the same language as the input.

Return ONLY a valid JSON array. No commentary, no markdown, just JSON.
```

**Step 2: Write the evaluator config/schema**

Create `prompts/competitor_evaluator/v1_config.json` — identical schema to researcher:

```json
{
  "json_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "required": ["name", "competitor_id", "aliases", "comparison_summary", "strengths", "angles", "switching_barriers", "anecdotes", "recent_news"],
      "additionalProperties": false,
      "properties": {
        "name": {
          "type": "string",
          "description": "Official company name as used in the market"
        },
        "competitor_id": {
          "type": "string",
          "description": "Stable snake_case identifier, e.g. 'norlys', 'andel_energi'"
        },
        "aliases": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Other names customers use: old brands, abbreviations, sub-brands"
        },
        "comparison_summary": {
          "type": "string",
          "description": "Complete pricing and terms picture covering ALL relevant dimensions, compared to our offering"
        },
        "strengths": {
          "type": "array",
          "items": { "type": "string" },
          "description": "What this competitor genuinely does well"
        },
        "angles": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Pre-computed comparative statements with specific numbers"
        },
        "switching_barriers": {
          "type": "array",
          "items": { "type": "string" },
          "description": "What holds customers: contractual, practical, psychological"
        },
        "anecdotes": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Concrete stories from reviews/forums with specific details"
        },
        "recent_news": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Dated items from last 12 months"
        }
      }
    }
  }
}
```

**Step 3: Commit**

```bash
git add prompts/competitor_evaluator/
git commit -m "feat: add competitor evaluator v1 prompt and schema"
```

---

### Task 3: Update `competitor_fetcher.py` — new helper functions

**Files:**
- Modify: `src/services/competitor_fetcher.py`

This task adds the new helper functions and updates existing ones. Does NOT change the service class yet.

**Step 1: Write tests for the new render function**

Add to `tests/test_competitor_fetcher.py`:

```python
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
    assert "De er gode til" not in result  # Section omitted when empty


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
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_competitor_fetcher.py -v -k "v2"
```

Expected: FAIL — `render_competitor_description` doesn't handle v2 fields yet.

**Step 3: Update `render_competitor_description` in `src/services/competitor_fetcher.py`**

Replace the function at lines 155-189 with a version that detects v2 vs v1 format:

```python
def render_competitor_description(research_data: dict) -> str:
    """Render competitor data as readable text for the state patcher's lookup_competitor tool.

    Supports both v2 format (comparison_summary, angles, strengths, etc.)
    and v1 format (overview, pricing_packages, pain_points, critical_news).
    Detects format by checking for the 'angles' key (v2) vs 'pricing_packages' key (v1).
    """
    # --- v2 format ---
    if "angles" in research_data or "comparison_summary" in research_data:
        return _render_v2(research_data)
    # --- v1 format (backwards compat) ---
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
```

**Step 4: Add `build_evaluator_prompt` helper function**

Add after `build_discovery_prompt` in `src/services/competitor_fetcher.py`:

```python
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
```

**Step 5: Run tests to verify they pass**

```bash
cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_competitor_fetcher.py -v
```

Expected: ALL PASS (including existing v1 tests which now go through `_render_v1`).

**Step 6: Commit**

```bash
git add src/services/competitor_fetcher.py tests/test_competitor_fetcher.py
git commit -m "feat: add v2 render function and evaluator prompt builder"
```

---

### Task 4: Update `CompetitorFetcherService` with researcher + evaluator pipeline

**Files:**
- Modify: `src/services/competitor_fetcher.py` (the `CompetitorFetcherService` class)

**Step 1: Write test for the new Gemini researcher call**

Add to `tests/test_competitor_fetcher.py`:

```python
def test_research_competitors_gemini_parses_json():
    """_research_competitors_gemini parses Gemini grounding response."""
    mock_response = MagicMock()
    mock_response.text = json.dumps([SAMPLE_V2_COMPETITOR])

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("services.competitor_fetcher.genai") as mock_genai:
        mock_genai.Client.return_value = mock_client
        from services.competitor_fetcher import _call_gemini_with_grounding
        result = _call_gemini_with_grounding(
            api_key="test-key",
            prompt="Research competitors",
        )

    assert len(result) == 1
    assert result[0]["name"] == "Norlys"
    assert "9 øre/kWh vs vores 4 øre" in result[0]["angles"][0]
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_competitor_fetcher.py::test_research_competitors_gemini_parses_json -v
```

Expected: FAIL — `_call_gemini_with_grounding` doesn't exist yet.

**Step 3: Refactor Gemini grounding call into reusable function**

In `src/services/competitor_fetcher.py`, rename `_discover_competitors_gemini` to `_call_gemini_with_grounding` and make it generic. Update the existing call site in `discover_competitors`:

```python
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

    competitors = json.loads(text.strip())
    if not isinstance(competitors, list):
        raise ValueError(f"Expected JSON array, got {type(competitors)}")

    return competitors


# Keep old name as alias for backwards compatibility with tests
_discover_competitors_gemini = _call_gemini_with_grounding
```

**Step 4: Add `research_competitors_v2` and `evaluate_competitors` methods to the service class**

Add to `CompetitorFetcherService`:

```python
def research_competitors_v2(
    self,
    campaign_info: str,
    custom_messages: str = "",
    source_urls: str = "",
    model: str = "gemini-2.5-pro",
) -> list[dict]:
    """Phase 1: Research competitors using Gemini + Google Search grounding.

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
    """Phase 2: Evaluate and improve researcher output.

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
```

**Step 5: Add `store_results_v2` method**

Add to `CompetitorFetcherService`:

```python
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
```

**Step 6: Add `generate_competitor_knowledge_v2` orchestration method**

Add to `CompetitorFetcherService`:

```python
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
```

**Step 7: Run all tests**

```bash
cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_competitor_fetcher.py tests/test_supabase_competitors.py -v
```

Expected: ALL PASS.

**Step 8: Commit**

```bash
git add src/services/competitor_fetcher.py tests/test_competitor_fetcher.py
git commit -m "feat: add v2 researcher+evaluator pipeline to CompetitorFetcherService"
```

---

### Task 5: Update the CLI (`run_competitor_fetcher.py`)

**Files:**
- Modify: `scripts/run_competitor_fetcher.py`

**Step 1: Update imports**

At the top of `scripts/run_competitor_fetcher.py`, the import from `competitor_fetcher` needs to also import any new display helpers. No new imports needed beyond what's already there — the service class handles everything.

**Step 2: Add v2 pipeline option to the main function**

After the campaign info collection (around line 252), add a pipeline version choice:

```python
# --- Pipeline version ---
pipeline = prompt_choice(
    "Which pipeline?",
    ["1 - v2 (Researcher + Evaluator, ~30-60s)", "2 - v1 (Discovery + Parallel AI, 5-25 min)"],
)
use_v2 = pipeline.startswith("1")
```

**Step 3: Add v2 flow after the choice**

Add a new branch for v2 that replaces Phase 1 + Phase 2 with the new pipeline:

```python
if use_v2:
    # --- v2: Researcher ---
    header("Phase 1: Researching Competitors (Gemini + Search)")
    t0 = time.perf_counter()
    try:
        competitors = service.research_competitors_v2(
            campaign_info=campaign_info,
            custom_messages=custom_messages,
            source_urls=source_urls,
        )
    except Exception as exc:
        error(f"Research failed: {exc}")
        sys.exit(1)
    success(f"Found {len(competitors)} competitors ({time.perf_counter() - t0:.1f}s)")

    # Save researcher output
    dump_dir = _PROJECT_ROOT / "customer_data" / campaign_id
    dump_dir.mkdir(parents=True, exist_ok=True)
    researcher_path = dump_dir / "competitor_researcher_output.json"
    _save_raw(researcher_path, competitors)

    # Display competitor names
    print()
    for i, comp in enumerate(competitors, 1):
        name = comp.get("name", "?")
        aliases = comp.get("aliases", [])
        alias_str = f" ({', '.join(aliases)})" if aliases else ""
        print(f"  {C.CYAN}{i}{C.RESET}) {C.BOLD}{name}{C.RESET}{alias_str}")
    print()

    # --- v2: Evaluator ---
    skip_eval = not prompt_confirm("Run evaluator to improve quality?")
    if not skip_eval:
        header("Phase 2: Evaluating & Improving (Gemini + Search)")
        t0 = time.perf_counter()
        try:
            competitors = service.evaluate_competitors(
                campaign_info=campaign_info,
                researcher_output=competitors,
            )
        except Exception as exc:
            error(f"Evaluation failed: {exc}")
            warn("Falling back to researcher output.")

        success(f"Evaluation complete ({time.perf_counter() - t0:.1f}s)")

    # Save final output
    final_path = dump_dir / "competitor_research_v2.json"
    _save_raw(final_path, competitors)

    # --- Display ---
    header("Results")
    for comp in competitors:
        from services.competitor_fetcher import render_competitor_description
        print(render_competitor_description(comp))
        print(f"\n{'─' * 40}\n")
    info(f"{len(competitors)} competitors researched")

    # --- Store ---
    if local_mode:
        success(f"Results saved to {final_path}")
    elif supabase_service:
        target = "local Supabase" if local_supabase else "Supabase"
        if prompt_confirm(f"Save to {target}?"):
            count = service.store_results_v2(campaign_id, competitors)
            success(f"Stored {count} competitors for '{campaign_id}' in {target}")

    header("Done!")
    return
```

Place this block right before the existing "Phase 1: Discovery" section, so the existing v1 flow becomes the else branch.

**Step 4: Make `parallel_api_key` optional in service init**

In `src/services/competitor_fetcher.py`, update `__init__` to not require `parallel_api_key` when using v2:

```python
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
```

And guard `research_competitors` (v1) to check `self.parallel_client is not None`.

**Step 5: Update CLI to not require PARALLEL_AI_API_KEY when using v2**

Move the parallel key check inside the v1 branch only:

```python
# Only require parallel key for v1 pipeline
if not use_v2:
    parallel_key = load_env_value("PARALLEL_AI_API_KEY")
    if not parallel_key:
        error("PARALLEL_AI_API_KEY is required for v1 pipeline.")
        sys.exit(1)
else:
    parallel_key = ""
```

**Step 6: Run the CLI manually to verify it works end-to-end**

```bash
cd /Users/mikkeldahl/callbuddy_service/scripts && python run_competitor_fetcher.py
```

Select v2 pipeline, use a known campaign ID, verify it produces output.

**Step 7: Commit**

```bash
git add scripts/run_competitor_fetcher.py src/services/competitor_fetcher.py
git commit -m "feat: add v2 pipeline option to competitor fetcher CLI"
```

---

### Task 6: Update `format_research_results_md` for v2 display

**Files:**
- Modify: `src/services/competitor_fetcher.py`
- Modify: `tests/test_competitor_fetcher.py`

**Step 1: Write test**

Add to `tests/test_competitor_fetcher.py`:

```python
def test_format_research_results_md_v2():
    """format_research_results_md handles v2 competitor list."""
    result = format_research_results_md({"competitors": [SAMPLE_V2_COMPETITOR]})
    assert "Norlys" in result
    assert "Vores fordele" in result or "angles" in result.lower() or "9 øre/kWh vs vores 4 øre" in result
```

**Step 2: Run test to see if it already passes**

```bash
cd /Users/mikkeldahl/callbuddy_service && python -m pytest tests/test_competitor_fetcher.py::test_format_research_results_md_v2 -v
```

If it fails, update `format_research_results_md` to handle v2 format (detect by checking for `angles` key). If it passes, this task is done.

**Step 3: Commit if changes were needed**

```bash
git add src/services/competitor_fetcher.py tests/test_competitor_fetcher.py
git commit -m "feat: update CLI display formatter for v2 competitor data"
```

---

### Task 7: End-to-end test with real campaign

**Files:** None (manual verification)

**Step 1: Run the full v2 pipeline with the existing ELG B2C campaign**

```bash
cd /Users/mikkeldahl/callbuddy_service/scripts && python run_competitor_fetcher.py
```

- Select "Local (JSON files)"
- Campaign ID: `elg_b2c` (or use the UUID from the existing test data)
- Load campaign info from Langfuse
- Select v2 pipeline
- Run evaluator: yes

**Step 2: Review the output**

Check `customer_data/<campaign_id>/competitor_research_v2.json`:
- Does every competitor have a complete `comparison_summary` with all pricing dimensions?
- Are `angles` referencing specific ELG numbers?
- Are `anecdotes` concrete stories, not generic complaints?
- Are `strengths` genuine things customers would value?
- Are `switching_barriers` realistic?
- Is `recent_news` dated and specific?

**Step 3: Test the lookup_competitor tool integration**

If Supabase is available, store the results and run the state patcher to verify `lookup_competitor` returns the new v2 rendered format:

```bash
cd /Users/mikkeldahl/callbuddy_service/scripts && python run_state_patcher.py --campaign-id <campaign_id>
```

Verify the rendered output matches the design doc format.

**Step 4: Commit any fixes**

```bash
git add -A && git commit -m "fix: adjustments from e2e testing of v2 competitor pipeline"
```
