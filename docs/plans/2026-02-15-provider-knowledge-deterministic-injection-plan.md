# Provider Knowledge Deterministic Injection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LLM-mediated competitor extraction with deterministic Supabase lookup triggered by a new `set_provider_knowledge` patch op.

**Architecture:** The state patcher LLM emits `set_provider_knowledge(provider_name="norlys")` when a customer mentions their provider. The orchestrator intercepts this patch, looks up the competitor brief from Supabase's `campaign_competitors` table, and writes the full description into `state.provider_knowledge`. The LLM never sees raw competitor data.

**Tech Stack:** Langfuse (prompt/config), Supabase (competitor data), Python (orchestrator)

**Design doc:** `docs/plans/2026-02-15-provider-knowledge-deterministic-injection-design.md`

---

### Task 1: Create v11 config with `set_provider_knowledge` op and `provider_knowledge` state field

**Files:**
- Create: `prompts/elg_b2c_state_patcher/v11_config.json`
- Reference: `prompts/elg_b2c_state_patcher/v10_config.json`

**Step 1: Create v11_config.json**

Copy `v10_config.json` and make two changes:

1. Add the `set_provider_knowledge` op to `json_schema.properties.patches.items.oneOf` array (after `update_customer_profile`):

```json
{
  "type": "object",
  "description": "Identify the customer's current provider from conversation. The system will automatically look up competitive intelligence. Free signal — does NOT count against the 0-2 patch budget. Only emit ONCE per conversation.",
  "required": ["op", "provider_name"],
  "additionalProperties": false,
  "properties": {
    "op": {
      "type": "string",
      "enum": ["set_provider_knowledge"]
    },
    "provider_name": {
      "type": "string",
      "enum": ["__DYNAMIC_COMPETITOR_ENUM__"],
      "description": "The competitor_id matching the customer's current provider."
    }
  }
}
```

The `"__DYNAMIC_COMPETITOR_ENUM__"` is a placeholder. The orchestrator replaces it at runtime with actual competitor IDs from Supabase. If no competitors are loaded, the entire `set_provider_knowledge` op is removed from the schema.

2. Add `"provider_knowledge": null` to `initial_state`:

```json
"initial_state": {
    "phase": "opener",
    "current_phase_notes": [],
    "phase_history": [],
    "customer_profile": null,
    "customer_facts": [],
    "pain_points": [],
    "objections": [],
    "value_props_delivered": [],
    "commitments": [],
    "provider_knowledge": null
}
```

**Step 2: Verify the JSON is valid**

Run: `python3 -c "import json; json.load(open('prompts/elg_b2c_state_patcher/v11_config.json')); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add prompts/elg_b2c_state_patcher/v11_config.json
git commit -m "feat(state-patcher): add set_provider_knowledge op to v11 config"
```

---

### Task 2: Create v11 prompt — replace competitor knowledge section with `set_provider_knowledge` instructions

**Files:**
- Create: `prompts/elg_b2c_state_patcher/v11_prompt.txt`
- Reference: `prompts/elg_b2c_state_patcher/v10_prompt.txt`

**Step 1: Create v11_prompt.txt**

Copy `v10_prompt.txt` and make these changes:

1. **Replace lines 532–540** (the "Konkurrent-baggrundsviden instruktion" section and `{{COMPETITOR_KNOWLEDGE}}`) with new `set_provider_knowledge` instructions:

```
## set_provider_knowledge

Når kunden nævner deres nuværende el-udbyder, emit `set_provider_knowledge` med det matchende `provider_name` fra enum-listen. Systemet slår automatisk konkurrent-baggrundsviden op og tilføjer det til state — du skal IKKE selv opsummere eller behandle det.

Felter: `op`, `provider_name`

Regler:
- Emit KUN én gang per samtale — når du først identificerer udbyderen.
- `provider_name` SKAL matche en værdi fra enum-listen i skemaet. Hvis kundens udbyder IKKE er i listen, emit IKKE denne patch (brug stadig `upsert_customer_fact(key="current_provider")` for alle udbydere).
- Tæller IKKE mod 0-2 patch-budgettet — det er et frit systemsignal.
- Emit typisk SAMMEN med `upsert_customer_fact(key="current_provider", value="<officielt navn>")`.

Eksempel: Kunden siger "Jeg er hos Norlys":
```json
{"op": "set_provider_knowledge", "provider_name": "norlys"}
{"op": "upsert_customer_fact", "key": "current_provider", "value": "Norlys", "source": "confirmed"}
```
```

2. **Remove the `{{COMPETITOR_KNOWLEDGE}}` line** at the very end of the file (after `</context>`). The new `set_provider_knowledge` section replaces it.

**Step 2: Verify line count is reasonable**

Run: `wc -l prompts/elg_b2c_state_patcher/v11_prompt.txt`
Expected: ~540 lines (similar to v10)

**Step 3: Commit**

```bash
git add prompts/elg_b2c_state_patcher/v11_prompt.txt
git commit -m "feat(state-patcher): replace competitor knowledge injection with set_provider_knowledge in v11 prompt"
```

---

### Task 3: Update orchestrator — add dynamic schema patching and Supabase competitor loading

**Files:**
- Modify: `scripts/run_state_patcher.py`

This task modifies the `main()` function to load competitors from Supabase at startup and patch the JSON schema with the real enum values.

**Step 1: Add Supabase import and initialization**

At the top of `main()` (after the Langfuse initialization around line 1277), add Supabase setup. Replace the old competitor prompt section (lines 1325–1334) with:

```python
    # Competitor knowledge — load competitor IDs from Supabase for provider_knowledge enum
    competitor_ids: List[str] = []
    campaign_id_input = prompt_string(
        "Supabase campaign_id (for competitor lookup, or blank to skip)",
        default="",
    )
    if campaign_id_input:
        try:
            from services.supabase_client import SupabaseService
            supabase_url = load_env_value("SUPABASE_URL")
            supabase_key = load_env_value("SUPABASE_SERVICE_ROLE_KEY")
            if supabase_url and supabase_key:
                sb = SupabaseService(supabase_url, supabase_key)
                competitors_raw = sb.get_campaign_competitor_ids(campaign_id_input)
                competitor_ids = [c["competitor_id"] for c in competitors_raw]
                if competitor_ids:
                    success(f"Loaded {len(competitor_ids)} competitor IDs: {competitor_ids}")
                else:
                    warn("No competitors found for this campaign in Supabase")
            else:
                warn("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY — skipping competitor loading")
        except Exception as exc:
            warn(f"Failed to load competitors from Supabase: {exc}")
```

**Step 2: Add schema patching function**

Add a new function near the top of the file (after `_find_customer_fact`, around line 371):

```python
def patch_schema_competitor_enum(
    json_schema: Dict[str, Any],
    competitor_ids: List[str],
) -> Dict[str, Any]:
    """Patch the JSON schema to inject real competitor IDs into set_provider_knowledge enum.

    If competitor_ids is empty, removes the set_provider_knowledge op entirely.
    Returns a deep copy with modifications applied.
    """
    schema = copy.deepcopy(json_schema)
    one_of = schema["properties"]["patches"]["items"]["oneOf"]

    provider_op_idx = None
    for i, op_schema in enumerate(one_of):
        props = op_schema.get("properties", {})
        op_enum = props.get("op", {}).get("enum", [])
        if "set_provider_knowledge" in op_enum:
            provider_op_idx = i
            break

    if provider_op_idx is None:
        return schema

    if not competitor_ids:
        # No competitors — remove the op entirely
        one_of.pop(provider_op_idx)
    else:
        # Replace placeholder enum with real IDs
        one_of[provider_op_idx]["properties"]["provider_name"]["enum"] = competitor_ids

    return schema
```

**Step 3: Wire schema patching into main()**

After loading competitors and before building `response_format` (around line 1307), add:

```python
    # Patch schema with real competitor enum (or remove op if no competitors)
    if competitor_ids or "__DYNAMIC_COMPETITOR_ENUM__" in json.dumps(json_schema):
        json_schema = patch_schema_competitor_enum(json_schema, competitor_ids)
        if competitor_ids:
            info(f"Patched schema: set_provider_knowledge enum = {competitor_ids}")
        else:
            info("Patched schema: removed set_provider_knowledge (no competitors)")
```

Move the `response_format` construction to AFTER this patching.

**Step 4: Store Supabase service and campaign_id for runtime lookups**

Keep references to `sb` and `campaign_id_input` so `apply_patches` can do lookups. The simplest approach: pass them into `run_experiment()` via a new optional parameter `supabase_lookup` (a callable).

Add a parameter to `run_experiment`:

```python
    provider_lookup: Optional[Callable[[str], Optional[str]]] = None,
```

And build the callable in main():

```python
    provider_lookup = None
    if competitor_ids and campaign_id_input:
        def _lookup_provider(provider_name: str) -> Optional[str]:
            return sb.get_competitor_description(campaign_id_input, provider_name)
        provider_lookup = _lookup_provider
```

Pass `provider_lookup=provider_lookup` in all three `run_experiment()` call sites (lines ~1385, ~1448, ~1520).

**Step 5: Commit**

```bash
git add scripts/run_state_patcher.py
git commit -m "feat(state-patcher): add Supabase competitor loading and dynamic schema patching"
```

---

### Task 4: Update `apply_patches` to handle `set_provider_knowledge` with deterministic Supabase lookup

**Files:**
- Modify: `scripts/run_state_patcher.py`

**Step 1: Update `apply_patches` signature and handler**

Add `provider_lookup` parameter to `apply_patches`:

```python
def apply_patches(
    state: Dict[str, Any],
    patch_list: List[dict],
    phase_checkpoints: Optional[Dict[str, List[str]]] = None,
    provider_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
```

Replace the existing `set_competitor_context` handler (lines 351–360) with `set_provider_knowledge`:

```python
        elif op == "set_provider_knowledge":
            provider_name = patch.get("provider_name")
            if not provider_name:
                warn("set_provider_knowledge: missing provider_name")
            elif state.get("provider_knowledge") is not None:
                warn(f"set_provider_knowledge: already set, ignoring duplicate for '{provider_name}'")
            elif provider_lookup is None:
                warn(f"set_provider_knowledge: no provider_lookup configured, skipping '{provider_name}'")
            else:
                description = provider_lookup(provider_name)
                if description:
                    state["provider_knowledge"] = {
                        "provider_name": provider_name,
                        "description": description,
                    }
                    info(f"Injected provider knowledge for '{provider_name}' ({len(description)} chars)")
                else:
                    state["provider_knowledge"] = {
                        "provider_name": provider_name,
                        "description": "",
                    }
                    warn(f"set_provider_knowledge: no data found for '{provider_name}'")
```

**Step 2: Pass `provider_lookup` through the call chain**

In `run_experiment()`, pass `provider_lookup` to `apply_patches`:

At line 1172, change:
```python
state = apply_patches(state, patches, phase_checkpoints=phase_checkpoints)
```
to:
```python
state = apply_patches(state, patches, phase_checkpoints=phase_checkpoints, provider_lookup=provider_lookup)
```

**Step 3: Remove old competitor injection code**

Delete these sections from `run_experiment()`:
- Lines 1091–1116: the `should_inject_competitor_knowledge` block and `COMPETITOR_KNOWLEDGE` variable setting
- Lines 1130–1158: the `use_tools` / `competitor_tool_service` block (kept for backwards compat but no longer needed)

Delete from `run_experiment()` signature:
- `competitor_tool_service` parameter
- `competitor_prompt_name` parameter

Add:
- `provider_lookup: Optional[Callable[[str], Optional[str]]] = None` parameter

Delete the standalone function:
- `should_inject_competitor_knowledge()` (lines 374–380)

Update all three call sites in `main()` to remove `competitor_tool_service` and `competitor_prompt_name`, and add `provider_lookup`.

**Step 4: Commit**

```bash
git add scripts/run_state_patcher.py
git commit -m "feat(state-patcher): handle set_provider_knowledge with deterministic Supabase lookup"
```

---

### Task 5: Push v11 prompt and config to Langfuse

**Files:**
- Reference: `prompts/elg_b2c_state_patcher/v11_prompt.txt`
- Reference: `prompts/elg_b2c_state_patcher/v11_config.json`

**Step 1: Push to Langfuse**

Write a small script to push the v11 prompt as a new version of the `elg_b2c_state_patcher` chat prompt in Langfuse. The prompt text goes into the system message, and the config (json_schema + initial_state) goes into the prompt config.

```python
import sys, json
sys.path.insert(0, 'src')
from services.langfuse import init_langfuse

lf = init_langfuse(push_to_langfuse=True)

with open('prompts/elg_b2c_state_patcher/v11_prompt.txt', 'r') as f:
    prompt_text = f.read()
with open('prompts/elg_b2c_state_patcher/v11_config.json', 'r') as f:
    config = json.load(f)

lf.create_prompt(
    name="elg_b2c_state_patcher",
    type="chat",
    prompt=[
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "## Current state\n\n{{current_state}}\n\n## Transcript\n\n#### Already accounted transcript\n{{old_transcript}}\n\n#### Latest transcript\n\n{{new_transcript}}"},
    ],
    config=config,
    labels=["latest"],
)
print("Pushed v11 to Langfuse")
```

Run: `python3 <script above>`

Do NOT add `"production"` label yet — that should be done manually after testing.

**Step 2: Verify in Langfuse**

Run: `python3 -c "import sys; sys.path.insert(0,'src'); from services.langfuse import init_langfuse, _with_rate_limit_backoff; lf=init_langfuse(push_to_langfuse=True); p=_with_rate_limit_backoff(lambda: lf.get_prompt(name='elg_b2c_state_patcher', type='chat', label='latest')); print(f'Version: {p.version}, Labels: {p.labels}, Config keys: {list(p.config.keys())}')"`

Expected: Version number incremented, labels include `latest`, config has `json_schema` and `initial_state`.

**Step 3: Commit** (no code changes, just verification)

---

### Task 6: Manual test — run experiment with v11

**No code changes.** Run the experiment script interactively:

```bash
cd scripts
python run_state_patcher.py
```

When prompted:
- Prompt name: `elg_b2c_state_patcher`
- Campaign path: `hello_sales/elg_b2c`
- Mode: `2` (single conversation)
- Supabase campaign_id: the UUID for elg_b2c campaign
- Select a conversation where the customer mentions a known competitor (Norlys, Andel Energi, etc.)

**Verify:**
1. At startup: "Loaded N competitor IDs" message appears
2. At startup: "Patched schema: set_provider_knowledge enum = [...]" message appears
3. During conversation: when customer mentions provider, LLM emits `set_provider_knowledge`
4. Immediately: "Injected provider knowledge for 'norlys' (N chars)" appears
5. State shows `provider_knowledge` populated with full competitor brief
