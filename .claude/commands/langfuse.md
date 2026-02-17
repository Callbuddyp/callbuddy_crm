---
description: Fetch, inspect, or discuss Langfuse prompts. Use for viewing prompt content, understanding composite prompt structure, or working on prompt versions.
argument-hint: <prompt-name> [version|label]
allowed-tools: [Bash, Read, Write, Edit, Glob, Grep, WebFetch]
---

# Langfuse Prompt Management

The user wants to work with Langfuse prompts. Parse `$ARGUMENTS` to determine the action.

## How to fetch prompts

Use Python via Bash to interact with Langfuse:

```python
import sys
sys.path.insert(0, 'src')
from services.langfuse import init_langfuse, _with_rate_limit_backoff

lf = init_langfuse(push_to_langfuse=True)

# Fetch text prompt (most prompts in this project are text type)
prompt = _with_rate_limit_backoff(lambda: lf.get_prompt(name="PROMPT_NAME", type="text"))

# Fetch chat prompt
prompt = _with_rate_limit_backoff(lambda: lf.get_prompt(name="PROMPT_NAME", type="chat"))

# Fetch specific version
prompt = _with_rate_limit_backoff(lambda: lf.get_prompt(name="PROMPT_NAME", type="text", version=6))

# Access prompt data
prompt.prompt      # The prompt text (for text type) or messages (for chat type)
prompt.config      # Dict with json_schema, initial_state, etc.
prompt.version     # Version number
prompt.labels      # Labels like ["production"]
```

## Composite prompts (Langfuse Composability)

Langfuse supports **prompt composability** — referencing other text prompts inside a prompt using tags:

```
@@@langfusePrompt:name=PromptName|label=production@@@
@@@langfusePrompt:name=PromptName|version=3@@@
```

When the SDK fetches and compiles the prompt, these tags are automatically replaced with the referenced prompt's content. This enables:

- **Modular prompt components** reused across multiple prompts (e.g., campaign info, system instructions)
- **Single source of truth** — update a base prompt and all dependents update automatically
- **Campaign-specific composition** — a responder prompt can reference a shared state patcher output format, campaign context, and strategy guidelines as separate prompts

In this project, composability is used in `src/services/prompt_generator.py`:
```python
f"@@@langfusePrompt:name={prompt_name}|label={label}@@@"
```

### Composite prompt architecture for CallBuddy

Each production prompt is assembled from parts:
- **Campaign context** — product info, competitor intelligence, closing scripts (per-campaign text prompt)
- **State patcher** — incremental conversation state builder (text prompt + json_schema config)
- **Responder** — real-time response generator that reads state + transcript (references campaign context via composability tag)
- **Analysis / Feedback / Summary** — post-call prompts that can also reference shared components

## Prompt version files

When iterating on prompts, save versions locally at:
```
prompts/<prompt_name>/v<N>_prompt.txt   — prompt text
prompts/<prompt_name>/v<N>_config.json  — json_schema + initial_state config
```

## Prompt discussion notes

Design discussions and draft prompts are stored at:
```
/Users/mikkeldahl/Notes/new_backend_discussion/
├── analysis_prompt/       — Layer 1 call analysis (phase_journey, objections, triggers, engagement)
├── feedback_prompt/       — Layer 2 feedback generation (13 sales parameters, scoring)
├── responder_prompt/      — Real-time response suggestions (v1 inline, v2 modular)
├── state_updater_prompt/  — Incremental state patching (DISC profiling, patch ops)
└── summary_prompt/        — Post-call summary (composite, campaign-configurable sections)
```

Also in the project repo:
```
new_backend_discussion/
├── case_identifier_prompt.md    — Evaluation case selection prompt
└── case_identifier_schema.json  — Schema for case identifier output
```

## Instructions

1. If the user provided a prompt name in $ARGUMENTS, fetch it from Langfuse and display the content
2. If the user wants to compare versions, fetch both and show a diff
3. If the user wants to save a version locally, save to `prompts/<name>/v<N>_prompt.txt` and `v<N>_config.json`
4. If the user wants to discuss or iterate on a prompt, read relevant notes from the discussion directories above
5. Always show: prompt name, version, type (text/chat), config keys, and content length before displaying full content
