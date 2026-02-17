# Provider Knowledge: Deterministic Injection via State Patcher

**Date:** 2026-02-15
**Status:** Approved

## Problem

The state patcher LLM refuses to reliably emit competitor knowledge as a `customer_fact`. The instruction `upsert_customer_fact(key="competitor_context", value="<summary>")` creates a semantic dissonance — competitor intelligence is not a customer fact. The LLM either ignores the instruction entirely or emits an empty value.

The current approach also requires injecting the full competitor brief into `{{COMPETITOR_KNOWLEDGE}}` for the LLM to extract from, adding prompt length and a second injection turn.

## Solution

Replace LLM-mediated competitor extraction with deterministic Supabase lookup. The state patcher's only job is to **identify the provider name** from conversation. The orchestrator handles the data injection.

## Design

### New patch operation: `set_provider_knowledge`

```json
{
  "op": "set_provider_knowledge",
  "provider_name": "norlys"
}
```

- `provider_name`: constrained to an enum of known `competitor_id` values from the `campaign_competitors` Supabase table
- Free signal — does NOT count against the 0-2 patch budget
- Emitted once when the customer mentions their provider

### New state field: `provider_knowledge`

```json
"provider_knowledge": null
```

After the orchestrator processes the patch:

```json
"provider_knowledge": {
  "provider_name": "norlys",
  "description": "<full competitor brief from Supabase>"
}
```

The `description` is filled deterministically by the orchestrator via `get_competitor_description(campaign_id, competitor_id)`, not by the LLM.

### Turn-by-turn flow

| Turn | What happens |
|------|-------------|
| N | Customer says "Jeg er hos Norlys" |
| N | State patcher emits `set_provider_knowledge(provider_name="norlys")` + `upsert_customer_fact(key="current_provider", value="Norlys")` |
| N | Orchestrator intercepts `set_provider_knowledge`, calls Supabase, writes full brief into `state.provider_knowledge` |
| N+1 | Responder sees `state.provider_knowledge` with complete competitor brief |

Single-turn resolution. No second injection pass needed.

### Enum loading

At experiment startup, fetch competitor IDs from Supabase via `get_campaign_competitor_ids(campaign_id)`. Inject the enum values into the JSON schema's `set_provider_knowledge.provider_name.enum` field dynamically before the experiment loop begins.

If no competitors exist for the campaign, omit the `set_provider_knowledge` op from the schema entirely.

## Changes

### What gets added
- `set_provider_knowledge` op in JSON schema (with dynamic enum from Supabase)
- `provider_knowledge: null` in `initial_state`
- Deterministic Supabase lookup in `apply_patches` handler
- Prompt instruction: when customer mentions provider, emit `set_provider_knowledge`

### What gets removed
- `{{COMPETITOR_KNOWLEDGE}}` template variable from the prompt
- `should_inject_competitor_knowledge()` function in `run_state_patcher.py`
- Runtime Langfuse competitor prompt fetch logic
- `upsert_customer_fact(key="competitor_context")` instruction in prompt
- "Konkurrent-baggrundsviden instruktion" section in prompt

### What stays
- `upsert_customer_fact(key="current_provider")` — still a valid customer fact
- `campaign_competitors` Supabase table and existing query methods
- `CompetitorToolService` class (still useful, reusable patterns)

## Files affected

| File | Change |
|------|--------|
| `prompts/elg_b2c_state_patcher/v11_prompt.txt` | New version: add `set_provider_knowledge` instructions, remove competitor knowledge injection section |
| `prompts/elg_b2c_state_patcher/v11_config.json` | Add `set_provider_knowledge` op to schema, add `provider_knowledge: null` to initial_state |
| `scripts/run_state_patcher.py` | Add `set_provider_knowledge` handler in `apply_patches`, load competitor enum at startup, remove `should_inject_competitor_knowledge()` and injection logic |
| Langfuse `elg_b2c_state_patcher` | Push v11 prompt + config as new version |
