# Assist Prompt Composition Design

**Date:** 2026-02-15
**Status:** Approved

## Problem

The three assist prompts (`objection_assistant`, `coach_assistant`, `script_assistant`) share significant common content — role definition, signal+tag protocol, state usage, deduplication logic, response principles, campaign data guidelines, and output format. Without composition, this content is duplicated across each prompt, making updates error-prone and inconsistent.

The existing manual action prompts (`elg_b2c_objection`, `elg_b2c_close`, etc.) already use Langfuse composability to share components. The assist prompts need the same treatment.

## Approach: Two-Layer Composition

One shared block (`assist_shared`) contains everything common. Each mode adds a decision block with its specific logic. Existing components (`output_markdown_suggestions`, `{campaign_id}_info`) are reused.

## Architecture

### Composition Diagram

```
┌─────────────────────────────────────────────────┐
│  objection_assistant (chat)                     │
│                                                 │
│  system:                                        │
│    ┌─────────────────────────────────────┐      │
│    │ assist_shared                       │      │
│    │  - Role                             │      │
│    │  - Signal+tag prefix format         │      │
│    │  - State usage                      │      │
│    │  - Previous_suggestions dedup       │      │
│    │  - Response principles              │      │
│    │  - Campaign data guide              │      │
│    └─────────────────────────────────────┘      │
│    ┌─────────────────────────────────────┐      │
│    │ {campaign_id}_info                  │      │
│    └─────────────────────────────────────┘      │
│    ┌─────────────────────────────────────┐      │
│    │ assist_objection_decision           │      │
│    │  - Valid tags: objection            │      │
│    │  - 3-step decision logic            │      │
│    └─────────────────────────────────────┘      │
│    ┌─────────────────────────────────────┐      │
│    │ output_markdown_suggestions         │      │
│    └─────────────────────────────────────┘      │
│                                                 │
│  user:                                          │
│    {{transcript}}                               │
│    <state>{{state}}</state>                     │
│    <previous_suggestions>                       │
│    {{previous_suggestions}}                     │
│    </previous_suggestions>                      │
└─────────────────────────────────────────────────┘
```

The same structure applies to `coach_assistant` and `script_assistant`, swapping only the decision block.

### Prompts to Create

| Prompt | Type | Approx size | Description |
|--------|------|-------------|-------------|
| `assist_shared` | text | ~2-3K chars | Shared role, signal format, state usage, dedup, response principles, campaign data guide |
| `assist_objection_decision` | text | ~2K chars | Objection classification, 3-step decision logic |
| `assist_coach_decision` | text | ~3K chars | Phase-aware coaching, 4 tags, intervention criteria |
| `assist_script_decision` | text | ~12-14K chars | Full sales strategy from ai_suggestion, always-show |
| `objection_assistant` | chat | composed | Overwrites existing v1 |
| `coach_assistant` | chat | composed | New |
| `script_assistant` | chat | composed | New |

### Existing Prompts Reused

| Prompt | Type | Used for |
|--------|------|----------|
| `output_markdown_suggestions` | text | Output format (utterance + description + response) |
| `{campaign_id}_info` | text | Campaign context (products, pricing, competitors) |

## Component Details

### `assist_shared`

Contains 6 sections, each in its own XML tag for clarity:

**1. Role (`<ROLLE>`)** — Establishes the assist pattern: fires on every CustomerTurnEnded, first task is to evaluate whether to show a suggestion. Frames response as a script the agent reads aloud. Core principles: empathic friend, read the emotion, one goal, give space, be authentic, match energy.

**2. Signal+tag prefix format (`<SIGNAL_FORMAT>`)** — First line of output MUST be `show|<tag>` or `skip`. If `show|<tag>`: followed by the response in output format. If `skip`: no further output. Valid tags depend on mode-specific instructions. Malformed format is silently dropped.

**3. State usage (`<STATE>`)** — How to read each state field: `phase` (current conversation phase), `customer_profile` (tone/style guide, follow approach_note), `customer_facts` (reference without re-asking), `pain_points` (build arguments on customer's own pain), `objections` (active/resolved with type and status — never repeat rejected approaches), `value_props_delivered` (don't re-pitch rejected props), `commitments` (use as momentum). Emphasizes state is from PREVIOUS turn, not updated with latest segment.

**4. Previous_suggestions dedup (`<PREVIOUS_SUGGESTIONS>`)** — List of suggestions already shown to the agent. If the same situation (or semantically identical variant) was already addressed → `skip`. Only `show` if the situation is new, escalated, or has new information. Don't waste the agent's attention with redundant suggestions.

**5. Response principles (`<SVAR_PRINCIPPER>`)** — Before responding, consider: what do they feel, what does state tell you about this customer, what's already been tried (check objections + previous_suggestions), what does customer_profile.approach_note say. Dos: acknowledge first, match severity, use customer's words, one argument per response, short sentences, double line breaks for 2+ sentences, same language as customer. Don'ts: repeat rejected approaches, multiple arguments (sounds desperate), minimize concerns, promise things contradicting campaign data, offer SMS/mail (only callback).

**6. Campaign data guide (`<KAMPAGNEDATA_GUIDE>`)** — Campaign data contains product facts, competitor intelligence, and sales strategies. Use product facts for grounding (never guess numbers). Use competitor intelligence strategically when customer mentions provider/binding/price. Never copy verbatim — reformulate to the situation and customer's language.

### `assist_objection_decision`

Extracted from the current `objection_assistant` v1. Contains the `<BESLUTNINGSLOGIK>` section:

**Valid tags:** `objection`

**3-step decision logic:**

1. **Is this an objection?** — Active resistance against the seller's proposal. Includes example lists of what IS an objection (CPR resistance, burned before, timing/binding, satisfied with current, skepticism, avoidance, blanket rejection) and what is NOT (clarifying questions, backchannel, low involvement, sharing facts, information seeking, silence). Key rule: a clarifying question is NOT an objection — "What does it cost?" is curiosity, "It's too expensive" is resistance.

2. **Already handled?** — Check `previous_suggestions` for semantic duplicates AND `objections` in state. Same objection rephrased → `skip`. Same objection but customer ESCALATED or added new information → `show|objection`. New objection → `show|objection`. Previously resolved objection resurfacing → `show|objection`.

3. **Generate response** — Only if step 1 = yes AND step 2 = new/escalated.

### `assist_coach_decision`

From the design notes in `prompt-coach.md`. Contains:

**Valid tags:** `objection`, `close`, `momentum`, `opener`

**Decision framework:** "Would an experienced sales manager write a note right now?" If no → `skip`.

**Phase-aware intervention criteria:**
- **Opener** → Missing identity/permission checkpoints → suggest permission question or hook (`opener` tag)
- **Discovery** → Missed pain/consequence opportunities → suggest SPIN Problem/Implication questions (`momentum` tag)
- **Pitch** → Value not connected to customer's pain → suggest bridge to pain point (`momentum` tag)
- **Close** → Customer is ready but no next step proposed → suggest concrete next step (`close` tag)
- **Objection handling** → Concern not acknowledged → suggest objection response (`objection` tag)

**Skip criteria:** Conversation flows naturally, agent just made a good move, only backchannel ("okay", "ja"), already coached on same situation (check previous_suggestions), inaudible/empty transcript.

**Rate limiting:** Max 1 active `momentum` suggestion at a time.

### `assist_script_decision`

Copied from the current `elg_b2c_ai_suggestion` system message (the `<INSTRUCTIONS>`, `<CONSTRAINTS>`, `<CONTEXT>`, `<REASONING>`, `<RECAP>` sections). Minus the role definition (in `assist_shared`), campaign data (composed via `{campaign_id}_info`), and output format (composed via `output_markdown_suggestions`).

**Valid tags:** `objection`, `close`, `momentum`, `opener`

**Decision logic:** Always `show|{tag}`. Never skip. Tag is determined by context:
- Customer raises objection → `objection`
- Closing/commitment opportunity → `close`
- Seller stuck or needs direction → `momentum`
- Opening/introduction phase → `opener`

**Strategy content (~12-14K chars):**
- Phase identification (opening / value / closing) with goals, success criteria, typical questions, transition triggers
- Customer state reading (neutral, resistance, curious, facts, ready) with signals and examples
- Resistance type classification (pre-emptive, experience-based, rational, practical, info-resistance, binding)
- Response tools (question, acknowledge, redirect, pitch, close, listen) with when-to-use
- Strategies: yes-ladder, benefit discovery, resistance escalation (max 3 attempts), assumptive closing, trial-period framing
- Reasoning steps before responding

## Composed Chat Prompts

All three follow the same template. Only the decision component reference differs.

### System message template

```
@@@langfusePrompt:name=assist_shared|label=production@@@

@@@langfusePrompt:name={campaign_id}_info|label=production@@@

@@@langfusePrompt:name=assist_{mode}_decision|label=production@@@

@@@langfusePrompt:name=output_markdown_suggestions|label=production@@@
```

Where `{mode}` is `objection_decision`, `coach_decision`, or `script_decision`.

### User message template

```
{{transcript}}

<state>
{{state}}
</state>

<previous_suggestions>
{{previous_suggestions}}
</previous_suggestions>
```

Same across all three prompts.

### Final prompt names

| Mode | Chat prompt name | Decision component |
|------|-----------------|-------------------|
| Mode 2 (objection) | `objection_assistant` | `assist_objection_decision` |
| Mode 3 (coach) | `coach_assistant` | `assist_coach_decision` |
| Mode 4 (liveAssist) | `script_assistant` | `assist_script_decision` |

## Future Work

- **Campaign-specific assist prompts**: `prompt_generator.py` creates composed campaign versions (e.g. `elg_b2c_objection_assistant`) via `config.json` `assist_actions` section. Same pattern as current manual prompts.
- **Splitting `assist_shared`**: If a specific concern (e.g. dedup rules) needs independent versioning, extract it from `assist_shared` into its own composable prompt. Non-breaking refactor.

## Implementation Order

1. Create `assist_shared` text prompt in Langfuse
2. Create `assist_objection_decision` text prompt (extract from current `objection_assistant`)
3. Create `assist_coach_decision` text prompt (from `prompt-coach.md` design notes)
4. Create `assist_script_decision` text prompt (copy from `elg_b2c_ai_suggestion` system message)
5. Create `objection_assistant` v2 as composed chat prompt (overwrites v1)
6. Create `coach_assistant` as composed chat prompt
7. Create `script_assistant` as composed chat prompt
8. Verify all three compose correctly via Langfuse SDK fetch + compile
