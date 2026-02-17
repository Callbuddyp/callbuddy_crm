# State Patcher v14 — Phase Notes + Objection Lifecycle

## Problem

Based on experiment results from `conv_1_incremental_v13` (Kimi K2 Thinking, Fireworks), three issues were identified:

1. **Objections not reflected in phase notes**: When an objection is raised, the model emits `add_objection` but almost never a corresponding `add_phase_progress` entry. The responder has no phase-level breadcrumb trail of objection events. Examples: steps 11, 14, 27, 30 — all missing phase progress.

2. **Objection graveyard**: Many objections are `raised` but never `addressed` or `resolved`. The model transitions away from `objection_handling` without closing out objection statuses. Examples: `gas_afvisning`, `ingen_besparelse`, `cub_rabat_ikke_40_kr`, `mistillid_til_reklame`.

3. **Soft objections missed**: Implicit timing/competition resistance like "Jeg har ikke sa lang tid siden jeg har skiftet" (step 6) is tracked only as `customer_fact`, not as an objection.

## Approach

Prompt-only changes + rename in JSON schema. No new patch operations. No client-side logic changes beyond field renaming.

## Changes

### 1. Rename `phase_progress` to `current_phase_notes`

- **State field**: `phase_progress` (array) -> `current_phase_notes`
- **Patch operation**: `add_phase_progress` -> `add_phase_note`
- **Scope expansion**: The field now captures milestones AND objection events, resolution attempts, and setbacks — any significant event in the current phase.

The existing milestone definitions per phase remain unchanged. A new category of notes is added: objection-linked notes.

### 2. Objection-to-note coupling rule

New rule added to the `add_objection` prompt section:

> When you emit `add_objection` or `update_objection`, you MUST also emit a corresponding `add_phase_note`. The note should capture the objection event in the context of the current phase.
>
> Examples:
> - `"objection_raised — Kunden afviser at skifte gas, har allerede hos OK -> salgsargument stoppet"`
> - `"objection_addressed — CPR-bekymring nedtonet via humor -> kunden aaben for forklaring"`

### 3. Objection lifecycle cleanup rule

New rule added to the `set_phase` prompt section:

> Before emitting `set_phase` to leave `objection_handling`, review all objections with status `raised` in current state. For each, emit `update_objection` to set status to `addressed`, `resolved`, or `resurfaced`. Do not leave objections in `raised` status when transitioning away from `objection_handling`.

### 4. Soft objection examples

Added to the `add_objection` section:

> Soft objections are also objections:
> - "Jeg har ikke sa lang tid siden jeg har skiftet" -> timing/competition
> - "Jeg synes det er uoverskueligt" -> need (overwhelmed)
> - "Kan du ikke sende det pa mail?" -> timing (deferral)

## Files to Change

1. **`prompts/elg_b2c_state_patcher/v8_prompt.txt`** — Copy to v9, apply all prompt changes
2. **`prompts/elg_b2c_state_patcher/v8_config.json`** — Copy to v9, update schema rename
3. **`scripts/run_state_patcher.py`** — Update `apply_patches` to use `current_phase_notes` and `add_phase_note`

## What Stays the Same

- All 10 patch operations remain (one renamed)
- JSON schema structure is identical (field/op name change only)
- Model's decision-making flow unchanged — coupling rules added, not new operations
- Phase definitions and milestone keys unchanged
- Client-side `apply_patches` logic unchanged beyond field name
