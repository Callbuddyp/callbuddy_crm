# Competitor Knowledge v2 — Design

## Problem

The current two-phase competitor pipeline (discovery + Parallel AI deep research) produces output that is:
- **Descriptive, not actionable** — factsheets instead of sales ammunition
- **Missing structured pricing** in the discovery phase, requiring expensive re-discovery
- **Vague on pain points** — "dårlig kundeservice" instead of concrete anecdotes
- **Slow and expensive** — Phase 2 takes 5-25 minutes and costs $0.10-0.30
- **Placeholder comparisons** — "vores X øre" because the research model lacks our pricing context

The responder needs suggestive, concrete information that naturally leads to good sales responses — not a competitive analysis document.

## Design Decisions

1. **The responder decides the angle** — competitor data provides ammunition, the responder chooses what to use based on conversation context.
2. **Pre-computed comparative statements** — comparisons to our offering are baked in so the responder doesn't have to do math/reasoning mid-call.
3. **Single enhanced pipeline** — replaces two-phase with a researcher + evaluator chain, both using Gemini + Google Search grounding. ~$0.02-0.06 and ~20-60 seconds total.
4. **Industry-agnostic** — no hardcoded pricing dimensions. The campaign info defines the market context; the model adapts.

## Pipeline Architecture

### Call 1: Researcher (Gemini + Google Search Grounding)

**Inputs:**
- `{{CAMPAIGN_INFO}}` — full campaign context including our pricing, products, target audience
- `{{CUSTOM_MESSAGES}}` — optional operator market knowledge
- `{{SOURCE_URLS}}` — optional reference URLs

**Task:**
1. Use Google Search to identify competitors a customer would likely mention
2. For each competitor, research current products, pricing (all dimensions), terms, reviews, and recent news
3. Compare everything against our offering from the campaign info
4. Write angles as pre-computed comparative statements a salesperson could use
5. Find concrete anecdotes — specific stories, not generic complaints
6. Identify what the competitor genuinely does well
7. Note switching barriers and recent changes

### Call 2: Evaluator (Gemini + Google Search Grounding)

**Inputs:**
- `{{CAMPAIGN_INFO}}` — same campaign context
- `{{RESEARCHER_OUTPUT}}` — full JSON from researcher

**Task:**
1. Check each competitor for missing pricing dimensions
2. Validate that comparisons to our offering are accurate against the campaign info
3. Sharpen vague angles — reject anything that wouldn't help a salesperson mid-call
4. Use Google Search to fill gaps (e.g., missed a recent price change)
5. Remove competitors that aren't relevant enough
6. Output the final amended JSON array

## Output Schema

```json
[
  {
    "name": "Norlys",
    "competitor_id": "norlys",
    "aliases": ["Eniig", "SE"],
    "comparison_summary": "Norlys tager 29 kr/md i abonnement og 9 øre/kWh i tillæg, ingen binding og 0 kr i oprettelse. Sammenlignet med vores spotaftale betaler kunden 10 kr mere i abonnement og 5 øre mere per kWh.",
    "strengths": [
      "Stærk lokal forankring som andelsejet selskab — mange kunder føler loyalitet.",
      "God app med live timepris-oversigt."
    ],
    "angles": [
      "Norlys' FlexEl har 9 øre/kWh i tillæg mod vores 4 øre — ved et gennemsnitligt forbrug på 4.000 kWh sparer kunden ca. 200 kr/år hos os.",
      "Deres abonnement er 29 kr/md uden nogen velkomstrabat — vi tilbyder de første 3 måneder gratis."
    ],
    "switching_barriers": [
      "Ingen binding, men mange kunder tror fejlagtigt de er bundet fordi de har været kunde længe.",
      "Kunder med Norlys Bredbånd kan frygte at miste samlerabat."
    ],
    "anecdotes": [
      "En kunde ventede 45 minutter i telefon for at opsige sin aftale hos Norlys, efter et estimat på 11 minutter. Hun skiftede til os dagen efter."
    ],
    "recent_news": [
      "Norlys hævede deres FlexEl-tillæg fra 7 til 9 øre/kWh i oktober 2025, hvilket ramte ca. 200.000 kunder."
    ]
  }
]
```

### Field descriptions

| Field | Purpose | Example quality bar |
|-------|---------|-------------------|
| `name` | Official company name | "Norlys" |
| `competitor_id` | Stable identifier for Supabase | "norlys" |
| `aliases` | Names customers use on the phone (old brands, abbreviations) | ["Eniig", "SE"] |
| `comparison_summary` | Complete pricing picture with ALL relevant dimensions, compared to us | Must cover every pricing metric relevant to this market |
| `strengths` | What they genuinely do well — responder acknowledges credibly | Not generic praise; specific things a customer would value |
| `angles` | Pre-computed comparative statements with our offering baked in | Must reference specific numbers and our product names |
| `switching_barriers` | What holds customers there (contractual, practical, psychological) | Actionable — helps responder anticipate and address resistance |
| `anecdotes` | Concrete stories from reviews/news that create doubt | Specific incidents, not "dårlig kundeservice" |
| `recent_news` | Price changes, service issues, press the customer may not know about | Dated, specific, verifiable |

## Integration with State Patcher

### What changes

- New researcher prompt (`prompts/competitor_researcher/v1_prompt.txt` + `v1_config.json`)
- New evaluator prompt (`prompts/competitor_evaluator/v1_prompt.txt` + `v1_config.json`)
- `render_competitor_description()` updated for new fields (strengths, angles, switching_barriers, anecdotes, recent_news)
- `competitor_fetcher.py` orchestrates researcher → evaluator instead of discovery → deep research
- Phase 2 (Parallel AI deep research) removed from default flow

### What stays the same

- Supabase schema (`campaign_competitors` table, `research_data` JSONB column)
- `lookup_competitor` tool in state patcher
- `set_competitor_context` patch operation
- CLI interface (`run_competitor_fetcher.py`)
- `get_competitor_description()` and `get_campaign_competitor_ids()` in supabase_client.py

### Rendered tool response format

When the state patcher calls `lookup_competitor`, it receives:

```
**Norlys** (også kendt som: Eniig, SE)

**Sammenligning:** Norlys tager 29 kr/md i abonnement og 9 øre/kWh i tillæg...

**De er gode til:**
- Stærk lokal forankring som andelsejet selskab...

**Vores fordele:**
- Norlys' FlexEl har 9 øre/kWh i tillæg mod vores 4 øre...

**Skiftbarrierer:**
- Ingen binding, men mange kunder tror fejlagtigt de er bundet...

**Historier:**
- En kunde ventede 45 minutter i telefon...

**Seneste nyt:**
- Norlys hævede deres FlexEl-tillæg fra 7 til 9 øre i oktober 2025...
```

The state patcher then distills what's relevant to the current conversation moment into `set_competitor_context.summary`.
