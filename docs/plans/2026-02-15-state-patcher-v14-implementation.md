# State Patcher v14 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve objection tracking in phase notes, fix objection lifecycle gaps, and catch soft objections — via prompt changes + field rename.

**Architecture:** Prompt-only behavioral changes (3 new rules + soft objection examples) plus a rename of `phase_progress` → `current_phase_notes` / `add_phase_progress` → `add_phase_note` in the JSON schema, initial state, prompt text, and `apply_patches` in the runner script.

**Tech Stack:** Langfuse text prompts (Danish), JSON Schema, Python (run_state_patcher.py)

**Design doc:** `docs/plans/2026-02-15-state-patcher-v14-design.md`

---

### Task 1: Create v9_config.json with renamed fields

**Files:**
- Source: `prompts/elg_b2c_state_patcher/v8_config.json`
- Create: `prompts/elg_b2c_state_patcher/v9_config.json`

**Step 1: Copy v8_config.json to v9_config.json**

```bash
cp prompts/elg_b2c_state_patcher/v8_config.json prompts/elg_b2c_state_patcher/v9_config.json
```

**Step 2: Apply renames in v9_config.json**

Three changes in the JSON:

1. In `json_schema.properties.patches.description`: change `add_phase_progress` → `add_phase_note`
2. In the `add_phase_progress` oneOf entry:
   - Change the `description` from "Record an achieved milestone in the current phase" to "Record a significant event in the current phase — milestones, objection events, resolution attempts, or setbacks. Free structural signal — does NOT count against the 0-2 patch budget. Item format: 'note_key — concise note: what happened → strategic signal'."
   - Change `op.enum` from `["add_phase_progress"]` to `["add_phase_note"]`
   - Change `item.description` from "Milestone key followed by dash..." to "Note key followed by dash and concise note (under ~80 chars). Format: 'note_key — fact → signal'."
3. In `initial_state`: change `"phase_progress": []` to `"current_phase_notes": []`

**Step 3: Verify JSON is valid**

Run: `python3 -c "import json; json.load(open('prompts/elg_b2c_state_patcher/v9_config.json'))"`

Expected: No output (valid JSON)

**Step 4: Commit**

```bash
git add prompts/elg_b2c_state_patcher/v9_config.json
git commit -m "feat(state-patcher): create v9 config with phase_progress → current_phase_notes rename"
```

---

### Task 2: Create v9_prompt.txt with all prompt changes

**Files:**
- Source: `prompts/elg_b2c_state_patcher/v8_prompt.txt`
- Create: `prompts/elg_b2c_state_patcher/v9_prompt.txt`

**Step 1: Copy v8_prompt.txt to v9_prompt.txt**

```bash
cp prompts/elg_b2c_state_patcher/v8_prompt.txt prompts/elg_b2c_state_patcher/v9_prompt.txt
```

**Step 2: Rename all occurrences of `phase_progress` → `current_phase_notes`**

Global replace in v9_prompt.txt:
- `phase_progress` → `current_phase_notes` (all occurrences)
- `add_phase_progress` → `add_phase_note` (all occurrences — both in op names and section headers)

**Step 3: Update the `add_phase_note` section header and description**

Change the section header from:

```
### add_phase_progress

Registrer en opnået milestone i den AKTUELLE fase. Milestones trackes inkrementelt og giver responderen real-time bevidsthed om hvad der er opnået — så den ikke gentager afsluttede trin og kan identificere mangler.
```

to:

```
### add_phase_note

Registrer en betydningsfuld hændelse i den AKTUELLE fase — milestones, indvendingshændelser, resolutionsforsøg, eller tilbageskridt. Notes trackes inkrementelt og giver responderen real-time bevidsthed om hvad der sker i fasen — så den ikke gentager afsluttede trin og kan identificere mangler.
```

**Step 4: Update the milestone section to include objection-linked notes**

After the existing milestone examples block (the four JSON examples), add this new section:

```
#### Indvendings-kobling

Når du emitter `add_objection` eller `update_objection`, SKAL du ALTID også emitte en tilsvarende `add_phase_note`. Noten skal fange indvendingshændelsen i konteksten af den aktuelle fase.

Eksempler:
```json
{"op": "add_phase_note", "item": "objection_raised — Kunden afviser at skifte gas, har allerede hos OK → salgsargument stoppet"}
{"op": "add_phase_note", "item": "objection_addressed — CPR-bekymring nedtonet via humor → kunden åben for forklaring"}
{"op": "add_phase_note", "item": "objection_resolved — Prisbekymring løst med konkret besparelsesbevis → kunden accepterede"}
```

This ensures the responder always sees objection events in the phase timeline.
```

**Step 5: Add objection lifecycle cleanup rule to `set_phase` section**

In the `set_phase` section, after the "SELVKONTROL FØR OUTPUT" subsection, add:

```
#### Indvendings-oprydning ved faseskift

Når du emitter `set_phase` for at FORLADE `objection_handling`, gennemgå ALLE indvendinger med status `raised` i current state. For HVER indvending, emit `update_objection` for at sætte status til `addressed`, `resolved` eller `resurfaced`. Efterlad IKKE indvendinger i `raised` status når du skifter væk fra `objection_handling`.
```

**Step 6: Add soft objection examples to `add_objection` section**

In the `add_objection` section, after the existing "VIGTIG: Når kunden nægter eller tøver med at give personlige oplysninger..." paragraph, add:

```
BLØDE INDVENDINGER: Ikke alle indvendinger er eksplicitte afvisninger. Disse er OGSÅ indvendinger:
- "Jeg har ikke så lang tid siden, jeg har skiftet" → timing/competition — kunden signalerer at skifte er unødvendigt
- "Jeg synes det er uoverskueligt" → need — kunden er overvældet og ikke klar til at engagere sig
- "Kan du ikke sende det på mail?" → timing — kunden udskyder beslutningen
- "Jeg har en samlet rabat" → competition — kunden bruger nuværende fordele som argument mod skifte
```

**Step 7: Update the `phase_progress` reference in the "HVAD EN GOD STATE GØR" section**

Change the line:
```
2. **Fasebevidsthed** — `phase` siger HVAD responderen skal gøre. `phase_progress` viser HVAD DER ER OPNÅET i denne fase (så den ikke gentager afsluttede trin). `phase_history` giver kontekst fra tidligere faser (manglende milestones → foreslå at gå tilbage, summaries → emotionel bue).
```

to:
```
2. **Fasebevidsthed** — `phase` siger HVAD responderen skal gøre. `current_phase_notes` viser HVAD DER SKER i denne fase — milestones, indvendinger, resolutioner (så responderen ikke gentager afsluttede trin og kan se indvendingsmønstre). `phase_history` giver kontekst fra tidligere faser (manglende milestones → foreslå at gå tilbage, summaries → emotionel bue).
```

**Step 8: Update the `set_phase` client-side effect description**

Change:
```
- Klient-side effekt: `state.phase` sættes til ny fase. Den gamle fases `{phase, progress, summary}` tilføjes til `phase_history`. `phase_progress` nulstilles til `[]`.
```

to:
```
- Klient-side effekt: `state.phase` sættes til ny fase. Den gamle fases `{phase, notes, summary}` tilføjes til `phase_history`. `current_phase_notes` nulstilles til `[]`.
```

**Step 9: Update the phase regression example**

Change:
```
phase_history: [
  {phase: "discovery", progress: ["current_situation — ..."], summary: "Initial discovery. Pain ikke identificeret."},
  {phase: "pitch", progress: [], summary: "Pitch forsøgt uden pain → kunden ikke engageret. Regression."}
]
// phase_progress trackes nu den genoptagne discovery
```

to:
```
phase_history: [
  {phase: "discovery", notes: ["current_situation — ..."], summary: "Initial discovery. Pain ikke identificeret."},
  {phase: "pitch", notes: [], summary: "Pitch forsøgt uden pain → kunden ikke engageret. Regression."}
]
// current_phase_notes trackes nu den genoptagne discovery
```

**Step 10: Update selektivitetsregler reference**

Change in rule 1:
```
`set_phase` og `add_phase_progress` tæller IKKE mod dette budget
```
to:
```
`set_phase` og `add_phase_note` tæller IKKE mod dette budget
```

**Step 11: Verify no remaining references to old names**

Run: `grep -c "phase_progress\|add_phase_progress" prompts/elg_b2c_state_patcher/v9_prompt.txt`

Expected: `0`

**Step 12: Commit**

```bash
git add prompts/elg_b2c_state_patcher/v9_prompt.txt
git commit -m "feat(state-patcher): create v9 prompt with phase notes, objection coupling, soft objection rules"
```

---

### Task 3: Update `apply_patches` in run_state_patcher.py

**Files:**
- Modify: `scripts/run_state_patcher.py:278-300`

**Step 1: Add `add_phase_note` as an accepted op alongside `add_phase_progress`**

At line 291, change:
```python
        elif op == "add_phase_progress":
```
to:
```python
        elif op in ("add_phase_progress", "add_phase_note"):
```

This keeps backwards compatibility with v8 configs while supporting v9.

**Step 2: Add `current_phase_notes` as an accepted field alongside `phase_progress`**

At line 279, change:
```python
            elif "phase_history" in state and "phase_progress" in state:
```
to:
```python
            elif "phase_history" in state and ("current_phase_notes" in state or "phase_progress" in state):
```

And update lines 283, 289, 299, 300 to use a helper that resolves the field name:

Before the `apply_patches` function (around line 259), add a small helper:

```python
def _phase_notes_key(state: Dict[str, Any]) -> str:
    """Return the state key for phase notes (supports both v8 and v9 schemas)."""
    if "current_phase_notes" in state:
        return "current_phase_notes"
    return "phase_progress"
```

Then update `apply_patches` to use it:

At line 279-289, change:
```python
            elif "phase_history" in state and "phase_progress" in state:
                old_phase = state.get("phase", "unknown")
                history_entry: Dict[str, Any] = {
                    "phase": old_phase,
                    "progress": list(state["phase_progress"]),
                }
                summary = patch.get("completed_phase_summary")
                if summary:
                    history_entry["summary"] = summary
                state["phase_history"].append(history_entry)
                state["phase_progress"] = []
```
to:
```python
            elif "phase_history" in state and ("current_phase_notes" in state or "phase_progress" in state):
                pn_key = _phase_notes_key(state)
                old_phase = state.get("phase", "unknown")
                history_entry: Dict[str, Any] = {
                    "phase": old_phase,
                    "notes": list(state[pn_key]),
                }
                summary = patch.get("completed_phase_summary")
                if summary:
                    history_entry["summary"] = summary
                state["phase_history"].append(history_entry)
                state[pn_key] = []
```

At lines 291-300, change:
```python
        elif op == "add_phase_progress":
            # v8 new format: key + note → fill checkpoint in phase_log
            if "phase_log" in state and state["phase_log"]:
                key = patch.get("key")
                note = patch.get("note")
                if key:
                    state["phase_log"][-1][key] = note
            # v8 old format: item string → append to phase_progress/phase_history
            elif "phase_progress" in state and patch.get("item"):
                state["phase_progress"].append(patch["item"])
```
to:
```python
        elif op in ("add_phase_progress", "add_phase_note"):
            # v8 new format: key + note → fill checkpoint in phase_log
            if "phase_log" in state and state["phase_log"]:
                key = patch.get("key")
                note = patch.get("note")
                if key:
                    state["phase_log"][-1][key] = note
            # item string → append to current_phase_notes / phase_progress
            else:
                pn_key = _phase_notes_key(state)
                if pn_key in state and patch.get("item"):
                    state[pn_key].append(patch["item"])
```

**Step 3: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('scripts/run_state_patcher.py').read()); print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/run_state_patcher.py
git commit -m "feat(state-patcher): support add_phase_note op and current_phase_notes field in apply_patches"
```

---

### Task 4: Run experiment to verify improvements

**Files:**
- No file changes — verification only

**Step 1: Run the state patcher experiment with the new v9 prompt/config**

Use `run_state_patcher.py` to run `conv_1` with the v9 prompt. The exact command depends on how the script selects prompt versions — check the CLI arguments or update the config path in the script.

**Step 2: Review the output**

Check the experiment log for:
1. `add_phase_note` ops appear alongside `add_objection` ops
2. Objections get `update_objection` before phase transitions away from `objection_handling`
3. Soft objections (like "har ikke så lang tid siden jeg har skiftet") are captured as objections
4. `current_phase_notes` appears in the state JSON instead of `phase_progress`

**Step 3: Compare with v13 baseline**

Compare against `customer_data/hello_sales/elg_b2c/state_experiments/conv_1_incremental_v13_20260214_174435.txt` to verify improvements at the specific steps that were problematic (steps 6, 11, 14, 27, 30, 43).
