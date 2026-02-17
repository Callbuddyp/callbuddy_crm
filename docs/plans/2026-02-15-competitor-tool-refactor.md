# Competitor Tool Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract competitor tool handling into a `CompetitorToolService`, stop pre-loading descriptions, and return brief confirmations to the LLM instead of full competitor text.

**Architecture:** New `src/services/competitor_tool.py` service owns init, tool definition building, and tool call handling. The LLM tool-calling functions (`call_llm_with_tools_openai`, `call_llm_with_tools`) become generic via a `tool_handler` callable. `run_state_patcher.py` delegates entirely to the service.

**Tech Stack:** Python, Supabase, OpenAI-compatible API (Fireworks), Google GenAI (Gemini/Vertex)

---

### Task 1: Create `CompetitorToolService`

**Files:**
- Create: `src/services/competitor_tool.py`

**Step 1: Write the service**

```python
"""Competitor tool service for on-demand competitor lookups during state patching."""

import time
from typing import Any, Callable, Dict, List, Optional

from services.supabase_client import SupabaseService


def _info(msg: str) -> None:
    print(f"\033[36m[info]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[33m[warn]\033[0m {msg}")


class CompetitorToolService:
    """Manages competitor tool definitions and handles tool calls.

    Fetches competitor IDs at init (lightweight), builds tool definitions,
    and handles on-demand lookups with state injection.
    """

    def __init__(self, supabase_service: SupabaseService, campaign_id: str) -> None:
        self._supabase = supabase_service
        self._campaign_id = campaign_id
        self._competitors: List[Dict[str, str]] = []

        t0 = time.perf_counter()
        self._competitors = supabase_service.get_campaign_competitor_ids(campaign_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _info(f"Fetched {len(self._competitors)} competitor IDs in {elapsed_ms:.0f}ms")

    @property
    def has_competitors(self) -> bool:
        return len(self._competitors) > 0

    def should_offer_tool(self, state: Dict[str, Any]) -> bool:
        """True if there are competitors not yet fetched into state."""
        if not self._competitors:
            return False
        backgrounds = state.get("competitor_backgrounds", {})
        return any(
            c["competitor_id"] not in backgrounds
            for c in self._competitors
        )

    def _get_name(self, competitor_id: str) -> str:
        """Get display name for a competitor_id."""
        for c in self._competitors:
            if c["competitor_id"] == competitor_id:
                return c.get("name", competitor_id)
        return competitor_id

    def build_tool_definition(self, provider: str) -> Any:
        """Build a tool declaration for lookup_competitor.

        Args:
            provider: "fireworks" returns OpenAI-style dict,
                      "gemini"/"vertex" returns a genai FunctionDeclaration.
        """
        enum_values = [c["competitor_id"] for c in self._competitors]

        if provider in ("fireworks", "fireworks-lite"):
            return {
                "type": "function",
                "function": {
                    "name": "lookup_competitor",
                    "description": (
                        "Look up detailed competitive intelligence for a specific competitor. "
                        "Call this when the customer mentions a competitor by name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "competitor_id": {
                                "type": "string",
                                "enum": enum_values,
                                "description": "The competitor to look up",
                            },
                        },
                        "required": ["competitor_id"],
                    },
                },
            }

        from google.genai import types

        return types.FunctionDeclaration(
            name="lookup_competitor",
            description=(
                "Look up detailed competitive intelligence for a specific competitor. "
                "Call this when the customer mentions a competitor by name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "competitor_id": {
                        "type": "string",
                        "enum": enum_values,
                        "description": "The competitor to look up",
                    },
                },
                "required": ["competitor_id"],
            },
        )

    def handle_tool_call(self, competitor_id: str, state: Dict[str, Any]) -> str:
        """Fetch competitor description from Supabase and inject into state.

        Returns a brief confirmation string (sent back to the LLM).
        """
        t0 = time.perf_counter()
        description = self._supabase.get_competitor_description(
            self._campaign_id, competitor_id
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if description is None:
            _warn(f"No data found for competitor '{competitor_id}' ({elapsed_ms:.0f}ms)")
            return f"No data found for competitor '{competitor_id}'."

        # Inject into state
        if "competitor_backgrounds" not in state:
            state["competitor_backgrounds"] = {}
        state["competitor_backgrounds"][competitor_id] = description

        name = self._get_name(competitor_id)
        _info(f"Fetched competitor '{competitor_id}' in {elapsed_ms:.0f}ms ({len(description)} chars)")
        return f"Added competitor background for {name}"
```

**Step 2: Commit**

```bash
git add src/services/competitor_tool.py
git commit -m "feat: add CompetitorToolService for on-demand competitor lookups"
```

---

### Task 2: Generalize `call_llm_with_tools_openai()` to use `tool_handler`

**Files:**
- Modify: `scripts/run_state_patcher.py:852-954` (the `call_llm_with_tools_openai` function)

**Step 1: Change the function signature**

Replace `supabase_service` and `campaign_id` params with `tool_handler`:

```python
@_retry_on_rate_limit
def call_llm_with_tools_openai(
    messages: List[Dict[str, str]],
    model: str,
    tool_definition: dict,
    tool_handler: Callable[[str], str],
    client_kwargs: Optional[Dict] = None,
    response_format: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Call an OpenAI-compatible API (Fireworks) with tool support.

    Args:
        tool_handler: callable that takes competitor_id and returns a brief
                      confirmation string. Side effects (state injection)
                      happen inside the handler.

    Returns {"content": str, "ttft_ms": float, "e2e_ms": float, "tool_calls": list}.
    """
```

**Step 2: Replace the Supabase fetch with `tool_handler` call**

In the tool call handling block (current lines 910-938), replace:

```python
        if message.tool_calls:
            tc = message.tool_calls[0]
            fc_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            competitor_id = fc_args.get("competitor_id", "")

            info(f"Tool call: lookup_competitor(competitor_id=\"{competitor_id}\")")

            # Query Supabase
            description = supabase_service.get_competitor_description(campaign_id, competitor_id)
            if description is None:
                description = f"No data found for competitor '{competitor_id}'."

            tool_calls_log.append({
                "name": tc.function.name,
                "args": fc_args,
                "response_length": len(description),
                "result": description,
            })

            # Append assistant message with tool call + tool response
            # Exclude None fields — Fireworks rejects extra nulls like refusal, annotations, etc.
            conversation.append(message.model_dump(exclude_none=True))
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": description,
            })
```

With:

```python
        if message.tool_calls:
            tc = message.tool_calls[0]
            fc_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            competitor_id = fc_args.get("competitor_id", "")

            info(f"Tool call: {tc.function.name}(competitor_id=\"{competitor_id}\")")

            # Delegate to handler — handles Supabase fetch + state injection
            tool_response = tool_handler(competitor_id)

            tool_calls_log.append({
                "name": tc.function.name,
                "args": fc_args,
                "response": tool_response,
            })

            # Append assistant message with tool call + tool response
            # Exclude None fields — Fireworks rejects extra nulls like refusal, annotations, etc.
            conversation.append(message.model_dump(exclude_none=True))
            conversation.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_response,
            })
```

**Step 3: Commit**

```bash
git add scripts/run_state_patcher.py
git commit -m "refactor: generalize call_llm_with_tools_openai to use tool_handler"
```

---

### Task 3: Generalize `call_llm_with_tools()` (Gemini/Vertex) to use `tool_handler`

**Files:**
- Modify: `scripts/run_state_patcher.py:707-849` (the `call_llm_with_tools` function)

**Step 1: Change the function signature**

Replace `supabase_service` and `campaign_id` params with `tool_handler`:

```python
@_retry_on_rate_limit
def call_llm_with_tools(
    messages: List[Dict[str, str]],
    model: str,
    tool_declaration,
    tool_handler: Callable[[str], str],
    client_kwargs: Optional[Dict] = None,
    response_format: Optional[Dict] = None,
    provider: str = "gemini",
) -> Dict[str, Any]:
    """Call Gemini/Vertex with tool support.

    Args:
        tool_handler: callable that takes competitor_id and returns a brief
                      confirmation string. Side effects (state injection)
                      happen inside the handler.

    Returns {"content": str, "ttft_ms": float, "e2e_ms": float, "tool_calls": list}.
    """
```

**Step 2: Replace the Supabase fetch with `tool_handler` call**

In the tool call handling block (current lines 795-833), replace:

```python
        if function_call_part:
            fc = function_call_part.function_call
            fc_args = dict(fc.args) if fc.args else {}
            competitor_id = fc_args.get("competitor_id", "")

            info(f"Tool call: lookup_competitor(competitor_id=\"{competitor_id}\")")

            # Query Supabase
            description = supabase_service.get_competitor_description(campaign_id, competitor_id)
            if description is None:
                description = f"No data found for competitor '{competitor_id}'."

            tool_calls_log.append({
                "name": fc.name,
                "args": fc_args,
                "response_length": len(description),
                "result": description,
            })

            # Append function call + response to contents for next turn
            contents.append(candidate.content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"description_md": description},
                            )
                        )
                    ],
                )
            )
```

With:

```python
        if function_call_part:
            fc = function_call_part.function_call
            fc_args = dict(fc.args) if fc.args else {}
            competitor_id = fc_args.get("competitor_id", "")

            info(f"Tool call: {fc.name}(competitor_id=\"{competitor_id}\")")

            # Delegate to handler — handles Supabase fetch + state injection
            tool_response = tool_handler(competitor_id)

            tool_calls_log.append({
                "name": fc.name,
                "args": fc_args,
                "response": tool_response,
            })

            # Append function call + response to contents for next turn
            contents.append(candidate.content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": tool_response},
                            )
                        )
                    ],
                )
            )
```

**Step 3: Commit**

```bash
git add scripts/run_state_patcher.py
git commit -m "refactor: generalize call_llm_with_tools to use tool_handler"
```

---

### Task 4: Update `run_experiment()` and the init block to use `CompetitorToolService`

**Files:**
- Modify: `scripts/run_state_patcher.py:1064-1085` (function signature)
- Modify: `scripts/run_state_patcher.py:1140-1198` (step loop tool logic)
- Modify: `scripts/run_state_patcher.py:1365-1403` (init block)
- Modify: all callers of `run_experiment()` (~lines 1440-1475, 1518-1540, 1595-1612)

**Step 1: Change `run_experiment()` signature**

Replace the three params `competitor_tool`, `supabase_service`, `campaign_id` with one:

```python
def run_experiment(
    langfuse_client,
    prompt_name: str,
    prompt_version: int,
    llm_provider: str,
    llm_model: str,
    llm_client_kwargs: Dict[str, Any],
    response_format: Dict[str, Any],
    initial_state: Dict[str, Any],
    utterance_indices: List[int],
    utterances: List[Utterance],
    labels: Dict[int, str],
    audio_name: str,
    campaign_context: str,
    output_dir: Path,
    run_name: str,
    model_label: str,
    phase_checkpoints: Optional[Dict[str, List[str]]] = None,
    competitor_tool_service: Optional["CompetitorToolService"] = None,
) -> List[dict]:
```

**Step 2: Update the step loop tool logic**

Replace lines 1140-1198 (the `use_tools` check, the branching LLM calls, and the post-hoc injection) with:

```python
        step_tool_calls: List[dict] = []

        try:
            use_tools = (
                competitor_tool_service is not None
                and competitor_tool_service.should_offer_tool(state)
                and llm_provider in _TOOL_SUPPORTED_PROVIDERS
            )
            if use_tools:
                tool_handler = lambda cid: competitor_tool_service.handle_tool_call(cid, state)
                tool_def = competitor_tool_service.build_tool_definition(llm_provider)
                if llm_provider in ("fireworks", "fireworks-lite"):
                    llm_result = call_llm_with_tools_openai(
                        messages,
                        model=llm_model,
                        tool_definition=tool_def,
                        tool_handler=tool_handler,
                        client_kwargs=llm_client_kwargs,
                        response_format=response_format,
                    )
                else:
                    llm_result = call_llm_with_tools(
                        messages,
                        model=llm_model,
                        tool_declaration=tool_def,
                        tool_handler=tool_handler,
                        client_kwargs=llm_client_kwargs,
                        response_format=response_format,
                        provider=llm_provider,
                    )
                step_tool_calls = llm_result.get("tool_calls", [])
            else:
                llm_result = call_llm(messages, provider=llm_provider, model=llm_model, client_kwargs=llm_client_kwargs, response_format=response_format)

            raw_response = llm_result["content"]
            ttft_ms = llm_result["ttft_ms"]
            e2e_ms = llm_result["e2e_ms"]

            info(f"LLM responded — TTFT: {ttft_ms:.0f}ms | E2E: {e2e_ms:.0f}ms")
            if step_tool_calls:
                info(f"Tool calls: {len(step_tool_calls)}")

            # Parse and apply patches
            patches = parse_patch_response(raw_response)
            state = apply_patches(state, patches, phase_checkpoints=phase_checkpoints)

            # No post-hoc injection needed — handle_tool_call already injected into state
```

**Step 3: Update init block (lines 1365-1403)**

Replace the entire pre-loading block with:

```python
    # Set up competitor tool service (if campaign_id provided)
    competitor_tool_service = None
    if campaign_id:
        try:
            from services.supabase_client import SupabaseService

            if use_local_supabase:
                sb_url = "http://127.0.0.1:54321"
                sb_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
            else:
                sb_url = load_env_value("SUPABASE_URL")
                sb_key = load_env_value("SUPABASE_SERVICE_ROLE_KEY")
            if sb_url and sb_key:
                from services.competitor_tool import CompetitorToolService

                supabase_service = SupabaseService(sb_url, sb_key)
                competitor_tool_service = CompetitorToolService(supabase_service, campaign_id)
                if competitor_tool_service.has_competitors:
                    info(f"Competitor tool ready")
                else:
                    info("No competitors found in Supabase")
                    competitor_tool_service = None
            else:
                warn("SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set — skipping competitor tool")
        except Exception as exc:
            warn(f"Could not set up competitor tool: {exc}")
```

**Step 4: Update all `run_experiment()` callers**

Replace `competitor_tool=competitor_tool, supabase_service=supabase_service, campaign_id=campaign_id` with `competitor_tool_service=competitor_tool_service` at all call sites (~3 locations).

**Step 5: Remove the now-unused `build_competitor_tool()` function**

Delete `scripts/run_state_patcher.py:652-704` — this logic now lives in `CompetitorToolService.build_tool_definition()`.

**Step 6: Commit**

```bash
git add scripts/run_state_patcher.py src/services/competitor_tool.py
git commit -m "refactor: wire CompetitorToolService into run_state_patcher"
```

---

### Task 5: Manual smoke test

**Step 1: Run the state patcher with a campaign that has competitors**

```bash
cd scripts
python run_state_patcher.py
```

Select a campaign with competitors (e.g. `elg_b2c`), provide the campaign ID, and run against a conversation where the customer mentions a competitor.

**Step 2: Verify the expected behavior**

Check the output for:
1. `[info] Fetched N competitor IDs in Xms` — at init
2. `competitor_backgrounds` is NOT in the initial state printout
3. When a competitor is mentioned: `Tool call: lookup_competitor(competitor_id="norlys")`
4. `[info] Fetched competitor 'norlys' in Xms (NNNN chars)`
5. `Added competitor background for Norlys` — brief response (not the full text)
6. On subsequent steps, `competitor_backgrounds.norlys` appears in the state

**Step 3: Commit the plan doc**

```bash
git add docs/plans/2026-02-15-competitor-tool-refactor.md
git commit -m "docs: add competitor tool refactor implementation plan"
```
