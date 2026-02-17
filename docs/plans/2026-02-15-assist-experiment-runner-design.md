# Assist Experiment Runner Design

**Date:** 2026-02-15
**Status:** Draft

## Problem

We need to evaluate assist prompts (objection, coach, script) end-to-end across full conversations, simulating how they behave in production. The key challenge is that assist prompts accumulate `previous_suggestions` to avoid duplication — we need to run them turn-by-turn to test this deduplication logic.

## Solution

A new script `scripts/run_assist_experiment.py` that walks through a conversation step-by-step using pre-computed state from `.state.json` files, calls a single assist prompt at each customer utterance, and accumulates `previous_suggestions` across turns. Outputs a single readable `.txt` log.

## Design Decisions

1. **One assist prompt at a time** — isolates evaluation per prompt type
2. **Pre-computed state** — uses `.state.json` from state patcher runs as ground truth, decoupling assist evaluation from state quality
3. **Step-aligned** — evaluates at each state patcher step boundary (customer utterances that triggered state changes)
4. **Langfuse prompts** — fetches prompt from Langfuse (same workflow as `run_experiments.py`)
5. **Local output only** — saves `.txt` log to `experiments/`, no Langfuse logging (fast iteration)

## Langfuse Prompts

Three assist prompts exist in Langfuse as chat prompts:

| Prompt | Version | Template Variables |
|--------|---------|-------------------|
| `elg_b2c_script_assist` | v2 | `new_transcript`, `state`, `previous_suggestions` |
| `elg_b2c_objection_assist` | v2 | `new_transcript`, `state`, `previous_suggestions` |
| `elg_b2c_coach_assist` | v1 | `transcript`, `state`, `previous_suggestions` |

Config: `{use_reasoning, use_internet, output_type}` — system messages are fully baked (no composability refs, campaign info already embedded).

## Data Flow

```
.state.json              conversation.json         Langfuse assist prompt
(pre-computed state)     (speaker-diarized)         (chat, compiled)
        │                       │                          │
        └───────────┬───────────┘                          │
                    ▼                                      │
         For each step in .state.json:                     │
         ┌─────────────────────────────┐                   │
         │ 1. state = prev step's      │                   │
         │    accumulated_state        │                   │
         │ 2. Build new_transcript:    │◄──────────────────┘
         │    last 5 utterances up to  │
         │    current utterance_index  │
         │ 3. Compile prompt with:     │
         │    {new_transcript, state,  │
         │     previous_suggestions}   │
         │ 4. Call LLM                 │
         │ 5. Parse show|<tag> / skip  │
         │ 6. If show: append output   │
         │    to previous_suggestions  │
         └─────────────────────────────┘
                    │
                    ▼
            experiments/<run>.txt
```

## Template Variable Construction

### `new_transcript` / `transcript`

Last N utterances (default N=5) up to and including the current step's `utterance_index`. Formatted with speaker labels from `speaker_map.json`:

```
Sælger:
  Goddag, goddag.

Kunden:
  Det er dig der har ringet til mig.

Sælger:
  Ja, det var det. Og så ringede du tilbage til mig.
```

Uses the same `speaker_label_map()` and formatting logic as `run_state_patcher.py`.

### `state`

JSON-serialized `accumulated_state` from the **previous** step. At step 0, uses `initial_state`. This matches production behavior where state hasn't been updated with the current segment yet.

### `previous_suggestions`

Accumulated text of all prior `show` outputs. After each step where the LLM outputs `show|<tag>`, the response body (everything after the first line) is appended. Format TBD — likely newline-separated with `[tag]` prefixes.

For `elg_b2c_script_assist`, this is passed but effectively unused per the prompt's design (it always shows).

## CLI Flow

Interactive, matching the pattern from `run_experiments.py`:

1. **Select assist prompt** — choose from discovered `*_assist` prompts in Langfuse, or type a name
2. **Select LLM model** — same model picker as `run_experiments.py`
3. **Select conversation** — pick `.state.json` from `customer_data/<firm>/<campaign>/state_experiments/`
4. **Confirm** — show summary, confirm before running
5. **Run** — iterate steps, stream output to console, save log

## Output Format

```
================================================================================
  ASSIST EXPERIMENT LOG
================================================================================
  Prompt:       elg_b2c_objection_assist (v2)
  Model:        Gemini 3 Flash (Minimal Thinking)
  Conversation: conv_1.mp3
  State source: conv_1_incremental_v9_20260215_140623.state.json
  Steps:        24
  Run:          gemini-3-flash_elg_b2c_objection_assist_v2_20260215_180000
================================================================================

────────────────────────────────────────────────────────────────────────────────
[Step 1/24 · Utterance 0→2 · Phase: opener · 340ms TTFT · 890ms E2E]

── Transcript (last 5 utterances) ──
Kunden:
  Hallo?

Sælger:
  Goddag, goddag.

Kunden:
  Det er dig der har ringet til mig.

── State (from prev step) ──
  phase: opener
  customer_profile: null
  objections: []
  pain_points: []

── Decision ──
skip

── Previous Suggestions (0 total) ──
(none)

────────────────────────────────────────────────────────────────────────────────
[Step 8/24 · Utterance 18→20 · Phase: discovery · 280ms TTFT · 1200ms E2E]

── Transcript (last 5 utterances) ──
Sælger:
  Og hvad betaler du så ca. om måneden?

Kunden:
  Jeg er ikke nøg for at oplyse CPR-nummeret.

── State (from prev step) ──
  phase: discovery
  customer_profile: {disc_primary: "C", approach_note: "Match hans ..."}
  objections: [{id: "trust_cpr", status: "raised"}]
  pain_points: [{id: "high_price", severity: "medium"}]

── Decision ──
show|objection

── Output ──
Det forstår jeg godt — det er noget personligt.

Kan jeg bare spørge: Hvad er det der bekymrer dig mest ved det?

── Previous Suggestions (1 total) ──
1. [objection] Det forstår jeg godt — det er noget personligt...
────────────────────────────────────────────────────────────────────────────────

================================================================================
  SUMMARY
================================================================================
  Steps:          24
  Shown:          6 (25%)
  Skipped:        18 (75%)
  Avg TTFT:       310ms
  Avg E2E:        950ms
  Tags: objection=4, close=1, momentum=1
================================================================================
```

## State Summary in Log

For readability, the log shows a compact state summary (not full JSON):
- `phase` — current phase
- `customer_profile` — DISC primary + approach_note (or null)
- `objections` — list of `{id, status}` pairs
- `pain_points` — list of `{id, severity}` pairs
- `value_props_delivered` — count only
- `commitments` — count only

## Reused Code

From existing scripts:
- **LLM calling**: Reuse `call_llm()` and `AVAILABLE_MODELS` from `run_experiments.py`
- **Prompt compilation**: Reuse `compile_prompt()` from `run_experiments.py`
- **Speaker mapping**: Reuse `load_speaker_map()`, `speaker_label_map()` from `run_state_patcher.py`
- **Conversation loading**: Reuse conversation JSON loading from `run_state_patcher.py`
- **ANSI colors / CLI helpers**: Reuse from `run_experiments.py`

## File Structure

```
scripts/
  run_assist_experiment.py    # New script
```

Single file. No new modules needed — reuses existing code via imports.
