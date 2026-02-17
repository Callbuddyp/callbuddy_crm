# Competitor Tool Refactor — Design

## Problem

The competitor tool currently pre-loads ALL competitor descriptions from Supabase into `competitor_backgrounds` at init time. This dumps thousands of tokens into every state message, the LLM never actually calls the tool (it's disabled when data is pre-loaded), and the tool response sends the full description back to the LLM unnecessarily.

### Current behavior (broken)

1. At init: fetch all competitor IDs → fetch all full descriptions → inject into `initial_state["competitor_backgrounds"]`
2. Tool is only offered when `competitor_backgrounds` is empty (never, because of step 1)
3. If tool IS called: full description sent back as tool response content AND injected into state

### Intended behavior

1. At init: fetch competitor IDs only (lightweight, for building tool definition)
2. Tool is always offered when there are unfetched competitors
3. LLM calls tool when customer mentions a competitor
4. Tool fetches from Supabase, injects into state as side effect, returns brief confirmation
5. LLM never sees the full competitor text — it's ready-to-use and needs no translation

## Design

### New service: `src/services/competitor_tool.py`

`CompetitorToolService` class that owns all competitor tool logic:

- `__init__(supabase_service, campaign_id)` — fetches competitor IDs with latency logging
- `build_tool_definition(provider)` — returns OpenAI-style or Gemini FunctionDeclaration with enum of available keys
- `handle_tool_call(competitor_id, state)` — fetches full description from Supabase, injects into `state["competitor_backgrounds"][competitor_id]`, returns `"Added competitor background for {name}"`
- `has_competitors` property — whether any competitors are available
- `should_offer_tool(state)` — true if there are competitors not yet in `state["competitor_backgrounds"]`

### Changes to LLM tool calling functions

`call_llm_with_tools_openai()` and `call_llm_with_tools()` become generic:

- Replace `supabase_service` + `campaign_id` params with a `tool_handler: Callable[[str], str]`
- Tool handler returns the string to send back to the LLM as tool response
- State mutation happens inside the handler (side effect via passed-by-reference state dict)
- `tool_calls` log still returned for tracing/TXT output

### Changes to `run_state_patcher.py`

**Init block:** Replace ~40 lines of pre-loading with:
```python
competitor_tool_service = CompetitorToolService(supabase_service, campaign_id)
```

**Step loop:** Replace inline tool logic with:
```python
use_tools = competitor_tool_service.should_offer_tool(state)
tool_handler = lambda cid: competitor_tool_service.handle_tool_call(cid, state)
```

Remove post-hoc state injection loop (now inside `handle_tool_call`).

### Data flow

```
Step N: Customer mentions "Norlys"
  1. LLM sees tool: lookup_competitor(enum: [norlys, ewii, ...])
  2. LLM calls lookup_competitor("norlys")
  3. handle_tool_call:
     - Fetches from Supabase (logged: "Fetched 'norlys' in 8ms, 2147 chars")
     - Injects state["competitor_backgrounds"]["norlys"] = full_description
     - Returns "Added competitor background for Norlys"
  4. LLM sees "Added competitor background for Norlys" → outputs patches

Step N+1: State (now including competitor_backgrounds.norlys) is in context
  - Responder can reference competitor data without state patcher processing it
```

### Latency logging

Two measurement points in `CompetitorToolService`:

1. `__init__`: "Fetched 6 competitor IDs in 12ms"
2. `handle_tool_call`: "Fetched competitor 'norlys' in 8ms (2,147 chars)"

### Scope

| File | Change |
|------|--------|
| `src/services/competitor_tool.py` | **New** — `CompetitorToolService` |
| `scripts/run_state_patcher.py` | **Simplify** — remove pre-loading, use service |
| `scripts/run_state_patcher.py` (tool functions) | **Generalize** — `tool_handler` callable |

No schema changes, no prompt changes, no Supabase table changes. Tool definition shape and state structure stay identical.

### Provider support

Design works identically across all providers (Fireworks, Gemini, Vertex). Fireworks is the first implementation target.
