# Assist Experiment Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `scripts/run_assist_experiment.py` — an interactive CLI that walks through a conversation turn-by-turn using pre-computed state, calls a single assist prompt from Langfuse at each customer utterance, accumulates `previous_suggestions`, and writes a readable `.txt` log.

**Architecture:** Single script that loads a `.state.json` + matching conversation JSON, iterates over state steps, builds transcript windows + state for each step, compiles the Langfuse assist prompt, calls the LLM, parses `show|<tag>` / `skip`, accumulates shown outputs, and writes everything to a local log file.

**Tech Stack:** Python 3.12, Langfuse SDK (prompts), existing `run_experiments.py` LLM calling patterns, existing `run_state_patcher.py` data loading helpers.

**Design doc:** `docs/plans/2026-02-15-assist-experiment-runner-design.md`

---

### Task 1: Script scaffold with imports and CLI helpers

**Files:**
- Create: `scripts/run_assist_experiment.py`

**Step 1: Create the script with boilerplate, imports, and ANSI color helpers**

```python
#!/usr/bin/env python3
"""
Interactive CLI for running assist prompt experiments across full conversations.

Walks through a conversation turn-by-turn using pre-computed state from
.state.json files, calls a single assist prompt at each customer utterance,
accumulates previous_suggestions, and writes a readable .txt log.

Usage:
    cd scripts
    python run_assist_experiment.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path setup – allow imports from src/
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
os.chdir(_PROJECT_ROOT)

from openai import OpenAI  # noqa: E402
from models.conversation import Conversation, Utterance  # noqa: E402
from services.langfuse import init_langfuse, _with_rate_limit_backoff  # noqa: E402
from services.llm_client import _retry_on_rate_limit  # noqa: E402
from utils import load_env_value  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI color helpers (same as run_experiments.py)
# ---------------------------------------------------------------------------

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"


def info(msg: str) -> None:
    print(f"{C.CYAN}[info]{C.RESET} {msg}")


def success(msg: str) -> None:
    print(f"{C.GREEN}[ok]{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}[warn]{C.RESET} {msg}")


def error(msg: str) -> None:
    print(f"{C.RED}[error]{C.RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{C.BOLD}{C.MAGENTA}{'=' * 50}")
    print(f"  {msg}")
    print(f"{'=' * 50}{C.RESET}\n")


def prompt_string(question: str, default: str = "", prefix: str = "") -> str:
    suffix = f" [{prefix}{default}]" if default else ""
    raw = input(f"{C.BOLD}{question}{suffix}: {C.RESET}").strip()
    return raw if raw else default


def prompt_confirm(question: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{C.BOLD}{question} ({hint}): {C.RESET}").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def prompt_choice(question: str, options: List[str]) -> int:
    """Present numbered options, return 0-based index of selected option."""
    print(f"\n{C.BOLD}{question}{C.RESET}")
    for i, opt in enumerate(options):
        print(f"  {C.CYAN}{i + 1}{C.RESET}) {opt}")
    while True:
        raw = input(f"{C.BOLD}Select [1-{len(options)}]: {C.RESET}").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        error(f"Invalid choice. Enter 1-{len(options)}.")
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): scaffold script with imports and CLI helpers"
```

---

### Task 2: Data loading helpers

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Add data loading functions**

These replicate the patterns from `run_state_patcher.py` lines 429-471 but are self-contained to avoid import coupling.

Add after the CLI helpers:

```python
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TRANSCRIPT_WINDOW = 5  # number of utterances to include in new_transcript


def load_state_file(path: Path) -> dict:
    """Load a .state.json file. Returns dict with 'conversation', 'initial_state', 'steps'."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_conversation(base_dir: Path, audio_stem: str) -> Conversation:
    """Load a processed conversation by audio_stem from base_dir/processed_conversations/."""
    conv_dir = base_dir / "processed_conversations"
    candidates = list(conv_dir.glob(f"{audio_stem}.json"))
    if not candidates:
        for f in sorted(conv_dir.glob("*.json")):
            with f.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if raw.get("audio_stem") == audio_stem:
                return Conversation.from_dict(raw.get("conversation", {}))
        raise FileNotFoundError(f"No processed conversation for {audio_stem} in {conv_dir}")
    with candidates[0].open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return Conversation.from_dict(raw.get("conversation", {}))


def load_speaker_map(base_dir: Path) -> dict:
    """Load speaker_map.json from campaign directory."""
    path = base_dir / "speaker_map.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def speaker_label_map(audio_stem: str, speaker_map: dict) -> Dict[int, str]:
    """Map speaker_id -> label ('Sælger' / 'Kunden')."""
    speaker1_is_seller = speaker_map.get(audio_stem)
    if not isinstance(speaker1_is_seller, bool):
        speaker1_is_seller = True
    return {
        1: "Sælger" if speaker1_is_seller else "Kunden",
        2: "Kunden" if speaker1_is_seller else "Sælger",
    }
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add data loading helpers"
```

---

### Task 3: Transcript formatting and variable building

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Add transcript formatting and variable building functions**

Add after the data loading section:

```python
# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def format_utterances(
    utterances: List[Utterance],
    start_idx: int,
    end_idx: int,
    labels: Dict[int, str],
) -> str:
    """Format utterances[start_idx:end_idx+1] with speaker labels.

    Merges consecutive same-speaker turns. Matches format used by
    run_state_patcher.py _format_utterances().
    """
    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, buffer
        if current_speaker is None or not buffer:
            return
        label = labels.get(current_speaker, f"Speaker {current_speaker}")
        parts.append(f"{label}:\n\n{' '.join(buffer)}")
        buffer = []

    for i in range(start_idx, min(end_idx + 1, len(utterances))):
        utt = utterances[i]
        text = utt.cleaned_text
        if not text:
            continue
        if utt.speaker_id == current_speaker:
            buffer.append(text)
            continue
        flush()
        current_speaker = utt.speaker_id
        buffer = [text]

    flush()
    return "\n\n".join(parts)


def build_transcript_window(
    utterances: List[Utterance],
    end_idx: int,
    labels: Dict[int, str],
    window: int = TRANSCRIPT_WINDOW,
) -> str:
    """Build a transcript window: the last `window` utterances up to end_idx (inclusive)."""
    start_idx = max(0, end_idx - window + 1)
    return format_utterances(utterances, start_idx, end_idx, labels)


# ---------------------------------------------------------------------------
# Template variable building
# ---------------------------------------------------------------------------

def build_template_variables(
    utterances: List[Utterance],
    current_utt_idx: int,
    state: dict,
    previous_suggestions: List[str],
    labels: Dict[int, str],
) -> Dict[str, str]:
    """Build the template variables dict for the assist prompt.

    Returns dict with keys matching the Langfuse prompt template:
    - new_transcript / transcript: last N utterances up to current_utt_idx
    - state: JSON-serialized accumulated state
    - previous_suggestions: accumulated show outputs
    """
    transcript = build_transcript_window(utterances, current_utt_idx, labels)
    state_str = json.dumps(state, ensure_ascii=False, indent=2)
    suggestions_str = "\n---\n".join(previous_suggestions) if previous_suggestions else "(none)"

    return {
        "new_transcript": transcript,
        "transcript": transcript,  # coach_assist uses this name
        "state": state_str,
        "previous_suggestions": suggestions_str,
    }
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add transcript formatting and variable building"
```

---

### Task 4: LLM calling and response parsing

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Add LLM calling (reuse pattern from run_experiments.py) and response parsing**

Add after the variable building section. The `call_llm` function follows the same pattern as `run_experiments.py` lines 203-307, and `parse_assist_response` handles the `show|<tag>` / `skip` first-line format.

```python
# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "1": {"provider": "groq", "model": "openai/gpt-oss-120b", "label": "GPT-OSS 120B (Groq)"},
    "2": {"provider": "groq", "model": "moonshotai/kimi-k2-instruct-0905", "label": "Kimi K2 Instruct (Groq)", "client_kwargs": {"reasoning_effort": None}},
    "3": {"provider": "gemini", "model": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Minimal Thinking)", "client_kwargs": {"thinking_level": "minimal"}},
    "4": {"provider": "gemini", "model": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Low Thinking)", "client_kwargs": {"thinking_level": "low"}},
    "5": {"provider": "fireworks", "model": "accounts/fireworks/models/kimi-k2-thinking", "label": "Kimi K2 Thinking (Fireworks)", "client_kwargs": {"max_tokens": 32768}},
    "6": {"provider": "fireworks", "model": "accounts/fireworks/models/kimi-k2-instruct-0905", "label": "Kimi K2 Instruct (Fireworks)"},
    "7": {"provider": "gemini-lite", "model": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Budget 1024 Thinking)"},
    "8": {"provider": "fireworks-lite", "model": "accounts/fireworks/models/kimi-k2-thinking", "label": "Kimi K2 Thinking (Fireworks, Budget 1024)"},
    "9": {"provider": "fireworks", "model": "accounts/fireworks/models/glm-4p7-flash", "label": "GLM-4p7 Flash (Fireworks)"},
}
DEFAULT_MODEL_KEY = "3"


@_retry_on_rate_limit
def call_llm(
    messages: List[Dict[str, str]],
    provider: str,
    model: str,
    client_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Call LLM and return content + timing metrics.

    Returns {"content": str, "ttft_ms": float, "e2e_ms": float}.
    Copied from run_experiments.py call_llm() — text response only (no json mode).
    """
    if provider == "groq":
        from groq import Groq

        api_key = load_env_value("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")

        groq_client = Groq(api_key=api_key)
        merged_kw = client_kwargs or {}
        kwargs: dict = dict(
            model=model,
            messages=list(messages),
            temperature=merged_kw.get("temperature", 1),
            max_completion_tokens=merged_kw.get("max_completion_tokens", 8192),
            top_p=merged_kw.get("top_p", 1),
            stream=True,
        )
        reasoning = merged_kw.get("reasoning_effort")
        if reasoning is not None:
            kwargs["reasoning_effort"] = reasoning

        t0 = time.perf_counter()
        stream = groq_client.chat.completions.create(**kwargs)
        ttft_ms: Optional[float] = None
        chunks: List[str] = []
        for chunk in stream:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            delta = chunk.choices[0].delta.content
            if delta:
                chunks.append(delta)
        e2e_ms = (time.perf_counter() - t0) * 1000
        content = "".join(chunks)
        if not content:
            raise ValueError("Groq returned no content")
        return {"content": content, "ttft_ms": ttft_ms or e2e_ms, "e2e_ms": e2e_ms}

    if provider in ("fireworks", "fireworks-lite"):
        api_key = load_env_value("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required")

        fw_client = OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
        merged_kw = client_kwargs or {}
        kwargs: dict = dict(
            model=model,
            messages=list(messages),
            temperature=merged_kw.get("temperature", 0.6),
            max_tokens=merged_kw.get("max_tokens", 4000),
            top_p=merged_kw.get("top_p", 1),
            presence_penalty=merged_kw.get("presence_penalty", 0),
            frequency_penalty=merged_kw.get("frequency_penalty", 0),
            extra_body={"top_k": merged_kw.get("top_k", 40)},
            stream=True,
        )

        t0 = time.perf_counter()
        stream = fw_client.chat.completions.create(**kwargs)
        ttft_ms: Optional[float] = None
        content_ttft_ms: Optional[float] = None
        chunks: List[str] = []
        for chunk in stream:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                if content_ttft_ms is None:
                    content_ttft_ms = (time.perf_counter() - t0) * 1000
                chunks.append(text)
        e2e_ms = (time.perf_counter() - t0) * 1000
        content = "".join(chunks)
        if not content:
            raise ValueError("Fireworks returned no content")
        return {"content": content, "ttft_ms": content_ttft_ms or ttft_ms or e2e_ms, "e2e_ms": e2e_ms}

    # Non-streaming fallback (Gemini, etc.)
    from services.llm_client import get_llm_client
    client = get_llm_client(provider, **(client_kwargs or {}))
    t0 = time.perf_counter()
    content = client.generate(messages=messages, model=model)
    e2e_ms = (time.perf_counter() - t0) * 1000
    return {"content": content, "ttft_ms": e2e_ms, "e2e_ms": e2e_ms}


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_assist_response(raw: str) -> Tuple[str, Optional[str], str]:
    """Parse an assist prompt response.

    Returns (decision, tag, body):
    - decision: "show" or "skip"
    - tag: the tag string (e.g. "objection") if show, else None
    - body: the response text after the first line (empty string for skip)
    """
    # Strip thinking tags from thinking models
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    lines = text.split("\n", 1)
    first_line = lines[0].strip().lower()

    if first_line == "skip":
        return "skip", None, ""

    match = re.match(r"show\|(\w+)", first_line)
    if match:
        tag = match.group(1)
        body = lines[1].strip() if len(lines) > 1 else ""
        return "show", tag, body

    # Fallback: if first line doesn't match expected format, treat entire response as show
    warn(f"Unexpected first line format: '{lines[0].strip()[:60]}' — treating as show|unknown")
    return "show", "unknown", text
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add LLM calling and response parsing"
```

---

### Task 5: Langfuse prompt compilation

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Add prompt compilation helper**

Add after the response parsing section. This mirrors `compile_prompt()` from `run_experiments.py` (lines 314-339).

```python
# ---------------------------------------------------------------------------
# Langfuse prompt helpers
# ---------------------------------------------------------------------------

def compile_prompt(
    langfuse_client, prompt_name: str, variables: Dict[str, str],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Fetch prompt from Langfuse, compile with variables, return (messages, config).

    Tries chat type first, falls back to text.
    """
    try:
        prompt = _with_rate_limit_backoff(
            lambda: langfuse_client.get_prompt(name=prompt_name, type="chat")
        )
        compiled = prompt.compile(**variables)
        if isinstance(compiled, list):
            return compiled, prompt.config or {}
    except Exception:
        pass

    prompt = _with_rate_limit_backoff(
        lambda: langfuse_client.get_prompt(name=prompt_name, type="text")
    )
    compiled = prompt.compile(**variables)
    return [{"role": "user", "content": str(compiled)}], prompt.config or {}
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add Langfuse prompt compilation"
```

---

### Task 6: Output formatting

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Add log formatting functions**

Add after the prompt helpers:

```python
# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_RULE = "=" * 80
_THIN_RULE = "\u2500" * 80


def fmt_state_summary(state: dict) -> str:
    """Format a compact state summary for the log."""
    parts = [f"  phase: {state.get('phase', '?')}"]

    cp = state.get("customer_profile")
    if cp:
        disc = cp.get("disc_primary", "?")
        note = cp.get("approach_note", "")
        note_preview = (note[:60] + "...") if len(note) > 60 else note
        parts.append(f"  customer_profile: {{disc_primary: \"{disc}\", approach_note: \"{note_preview}\"}}")
    else:
        parts.append("  customer_profile: null")

    objs = state.get("objections", [])
    if objs:
        obj_strs = [f"{{id: \"{o.get('id', '?')}\", status: \"{o.get('status', '?')}\"}}" for o in objs]
        parts.append(f"  objections: [{', '.join(obj_strs)}]")
    else:
        parts.append("  objections: []")

    pains = state.get("pain_points", [])
    if pains:
        pain_strs = [f"{{id: \"{p.get('id', '?')}\", severity: \"{p.get('severity', '?')}\"}}" for p in pains]
        parts.append(f"  pain_points: [{', '.join(pain_strs)}]")
    else:
        parts.append("  pain_points: []")

    vp_count = len(state.get("value_props_delivered", []))
    parts.append(f"  value_props_delivered: {vp_count}")

    commit_count = len(state.get("commitments", []))
    parts.append(f"  commitments: {commit_count}")

    return "\n".join(parts)


def fmt_suggestions_list(suggestions: List[Tuple[str, str]]) -> str:
    """Format the running previous_suggestions list. Each entry is (tag, body_preview)."""
    if not suggestions:
        return "(none)"
    lines = []
    for i, (tag, body) in enumerate(suggestions):
        preview = body.replace("\n", " ")[:80]
        if len(body) > 80:
            preview += "..."
        lines.append(f"{i + 1}. [{tag}] {preview}")
    return "\n".join(lines)


def fmt_file_header(
    prompt_name: str,
    prompt_version: int,
    model_label: str,
    conversation: str,
    state_source: str,
    num_steps: int,
    run_name: str,
) -> str:
    return (
        f"{_RULE}\n"
        f"  ASSIST EXPERIMENT LOG\n"
        f"{_RULE}\n"
        f"  Prompt:       {prompt_name} (v{prompt_version})\n"
        f"  Model:        {model_label}\n"
        f"  Conversation: {conversation}\n"
        f"  State source: {state_source}\n"
        f"  Steps:        {num_steps}\n"
        f"  Run:          {run_name}\n"
        f"{_RULE}\n"
    )


def fmt_step_block(
    step_num: int,
    total_steps: int,
    prev_utt_idx: int,
    curr_utt_idx: int,
    phase: str,
    ttft_ms: float,
    e2e_ms: float,
    transcript: str,
    state_summary: str,
    decision: str,
    tag: Optional[str],
    body: str,
    suggestions_display: str,
) -> str:
    decision_line = f"show|{tag}" if decision == "show" else "skip"
    output_section = ""
    if decision == "show" and body:
        output_section = f"\n\u2500\u2500 Output \u2500\u2500\n{body}\n"

    return (
        f"\n{_THIN_RULE}\n"
        f"[Step {step_num}/{total_steps} \u00b7 Utterance {prev_utt_idx}\u2192{curr_utt_idx}"
        f" \u00b7 Phase: {phase} \u00b7 {ttft_ms:.0f}ms TTFT \u00b7 {e2e_ms:.0f}ms E2E]\n\n"
        f"\u2500\u2500 Transcript (last {TRANSCRIPT_WINDOW} utterances) \u2500\u2500\n"
        f"{transcript}\n\n"
        f"\u2500\u2500 State (from prev step) \u2500\u2500\n"
        f"{state_summary}\n\n"
        f"\u2500\u2500 Decision \u2500\u2500\n"
        f"{decision_line}\n"
        f"{output_section}\n"
        f"\u2500\u2500 Previous Suggestions ({len(suggestions_display.splitlines()) if suggestions_display != '(none)' else 0} total) \u2500\u2500\n"
        f"{suggestions_display}\n"
        f"{_THIN_RULE}\n"
    )


def fmt_footer(
    total_steps: int,
    shown: int,
    skipped: int,
    avg_ttft: float,
    avg_e2e: float,
    tag_counts: Dict[str, int],
) -> str:
    tag_str = ", ".join(f"{t}={c}" for t, c in sorted(tag_counts.items())) if tag_counts else "(none)"
    show_pct = int(100 * shown / total_steps) if total_steps else 0
    skip_pct = int(100 * skipped / total_steps) if total_steps else 0
    return (
        f"\n{_RULE}\n"
        f"  SUMMARY\n"
        f"{_RULE}\n"
        f"  Steps:        {total_steps}\n"
        f"  Shown:        {shown} ({show_pct}%)\n"
        f"  Skipped:      {skipped} ({skip_pct}%)\n"
        f"  Avg TTFT:     {avg_ttft:.0f}ms\n"
        f"  Avg E2E:      {avg_e2e:.0f}ms\n"
        f"  Tags:         {tag_str}\n"
        f"{_RULE}\n"
    )
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add output formatting functions"
```

---

### Task 7: Main CLI — setup and configuration

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Add the main function with CLI setup (prompt selection, model selection, state file selection)**

Add the main function:

```python
# ---------------------------------------------------------------------------
# State file discovery
# ---------------------------------------------------------------------------

def discover_state_files(search_dir: Path) -> List[Path]:
    """Find all .state.json files under search_dir recursively."""
    return sorted(search_dir.rglob("*.state.json"))


def infer_campaign_dir(state_path: Path) -> Path:
    """Infer the campaign base directory from a .state.json path.

    Expects structure: customer_data/<firm>/<campaign>/state_experiments/<file>.state.json
    Returns the campaign dir (2 levels up from state_experiments/).
    """
    # Walk up until we find processed_conversations/ sibling
    candidate = state_path.parent.parent
    if (candidate / "processed_conversations").exists():
        return candidate
    # Fallback: try parent
    candidate = state_path.parent
    if (candidate / "processed_conversations").exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find processed_conversations/ relative to {state_path}. "
        f"Expected structure: customer_data/<firm>/<campaign>/state_experiments/<file>.state.json"
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    header("Assist Experiment Runner")

    # 1. Init Langfuse
    langfuse_client = init_langfuse(push_to_langfuse=True)
    if langfuse_client is None:
        error("Could not initialize Langfuse. Check your .env credentials.")
        sys.exit(1)
    success("Langfuse initialized")

    # 2. Prompt name
    prompt_name = prompt_string("Assist prompt name (from Langfuse)", default="elg_b2c_objection_assist")
    if not prompt_name:
        error("Prompt name is required.")
        sys.exit(1)

    info(f"Fetching prompt '{prompt_name}'...")
    try:
        prompt_obj = _with_rate_limit_backoff(
            lambda: langfuse_client.get_prompt(name=prompt_name, type="chat")
        )
    except Exception:
        try:
            prompt_obj = _with_rate_limit_backoff(
                lambda: langfuse_client.get_prompt(name=prompt_name, type="text")
            )
        except Exception as exc:
            error(f"Failed to fetch prompt '{prompt_name}': {exc}")
            sys.exit(1)
    prompt_version = prompt_obj.version
    success(f"Prompt '{prompt_name}' v{prompt_version}")

    # 3. Model selection
    print(f"\n{C.BOLD}Available models:{C.RESET}")
    for key, entry in AVAILABLE_MODELS.items():
        marker = " (default)" if key == DEFAULT_MODEL_KEY else ""
        print(f"  {C.CYAN}{key}{C.RESET}) {entry['label']}{marker}")
    model_choice = prompt_string("Select model", default=DEFAULT_MODEL_KEY)
    if model_choice not in AVAILABLE_MODELS:
        error(f"Invalid choice '{model_choice}'. Using default.")
        model_choice = DEFAULT_MODEL_KEY
    selected = AVAILABLE_MODELS[model_choice]
    llm_provider = selected["provider"]
    llm_model = selected["model"]
    llm_client_kwargs = selected.get("client_kwargs", {})
    model_label = selected["label"]
    success(f"Model: {model_label}")

    # 4. State file selection
    customer_data_dir = _PROJECT_ROOT / "customer_data"
    state_files = discover_state_files(customer_data_dir)
    if not state_files:
        error(f"No .state.json files found under {customer_data_dir}")
        sys.exit(1)

    # Show relative paths for readability
    state_options = [str(f.relative_to(_PROJECT_ROOT)) for f in state_files]
    state_idx = prompt_choice("Select state file", state_options)
    state_path = state_files[state_idx]
    success(f"State file: {state_path.name}")

    # 5. Load state data
    info("Loading state data...")
    state_data = load_state_file(state_path)
    audio_ref = state_data.get("conversation", "unknown")
    audio_stem = audio_ref.rsplit(".", 1)[0] if "." in audio_ref else audio_ref
    initial_state = state_data.get("initial_state", {})
    steps = state_data.get("steps", [])
    if not steps:
        error("State file has no steps.")
        sys.exit(1)
    success(f"Loaded {len(steps)} steps for {audio_ref}")

    # 6. Load conversation
    campaign_dir = infer_campaign_dir(state_path)
    info(f"Loading conversation from {campaign_dir.relative_to(_PROJECT_ROOT)}/processed_conversations/...")
    conversation = load_conversation(campaign_dir, audio_stem)
    utterances = conversation.utterances
    success(f"Loaded {len(utterances)} utterances")

    # 7. Load speaker map
    speaker_map = load_speaker_map(campaign_dir)
    labels = speaker_label_map(audio_stem, speaker_map)
    info(f"Speaker labels: {labels}")

    # 8. Run name
    model_slug = model_label.lower().replace(" ", "-").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run = f"{model_slug}_{prompt_name}_v{prompt_version}_{timestamp}"
    run_name = prompt_string("Run name", default=default_run)

    # 9. Summary + confirm
    header("Experiment Summary")
    print(f"  Prompt:       {C.CYAN}{prompt_name}{C.RESET} (v{prompt_version})")
    print(f"  Model:        {C.CYAN}{model_label} ({llm_model}){C.RESET}")
    print(f"  Conversation: {C.CYAN}{audio_ref}{C.RESET} ({len(utterances)} utterances)")
    print(f"  State source: {C.CYAN}{state_path.name}{C.RESET} ({len(steps)} steps)")
    print(f"  Run name:     {C.CYAN}{run_name}{C.RESET}")
    print()

    if not prompt_confirm("Proceed?"):
        info("Aborted.")
        sys.exit(0)

    # ... (experiment loop in next task)


if __name__ == "__main__":
    main()
```

**Step 2: Run the script to verify CLI setup works (it should display options and exit after confirmation)**

```bash
cd /Users/mikkeldahl/callbuddy_service/scripts && python run_assist_experiment.py
```

Manually verify: prompt shows, model selection works, state files are discovered, summary displays. Ctrl+C to exit.

**Step 3: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add main CLI setup and configuration"
```

---

### Task 8: Main experiment loop

**Files:**
- Modify: `scripts/run_assist_experiment.py`

**Step 1: Replace the `# ... (experiment loop in next task)` comment in `main()` with the actual experiment loop**

Add this code in the main function, replacing the placeholder comment:

```python
    # 10. Open local file
    experiments_dir = _PROJECT_ROOT / "experiments"
    experiments_dir.mkdir(exist_ok=True)
    local_path = experiments_dir / f"{run_name}.txt"
    local_file = open(local_path, "w", encoding="utf-8")
    local_file.write(fmt_file_header(
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        model_label=f"{model_label} ({llm_model})",
        conversation=audio_ref,
        state_source=state_path.name,
        num_steps=len(steps),
        run_name=run_name,
    ))
    local_file.flush()
    success(f"Saving results to {local_path}")

    # 11. Run experiment loop
    header("Running Experiment")

    previous_suggestions: List[str] = []  # raw body text of each show output
    suggestions_display: List[Tuple[str, str]] = []  # (tag, body) for display
    total_ttft_ms = 0.0
    total_e2e_ms = 0.0
    completed = 0
    failed = 0
    shown = 0
    skipped = 0
    tag_counts: Dict[str, int] = {}

    for step_idx, step in enumerate(steps):
        curr_utt_idx = step["utterance_index"]
        # State from previous step (or initial_state for step 0)
        if step_idx == 0:
            state = initial_state
        else:
            state = steps[step_idx - 1]["accumulated_state"]

        # Previous utterance index (for display)
        prev_utt_idx = steps[step_idx - 1]["utterance_index"] if step_idx > 0 else 0
        phase = state.get("phase", "?")

        info(f"Step {step_idx + 1}/{len(steps)} · Utterance {prev_utt_idx}→{curr_utt_idx} · Phase: {phase}")

        # Build template variables
        variables = build_template_variables(
            utterances=utterances,
            current_utt_idx=curr_utt_idx,
            state=state,
            previous_suggestions=previous_suggestions,
            labels=labels,
        )

        transcript_display = variables["new_transcript"]
        state_summary = fmt_state_summary(state)

        # Compile prompt
        try:
            messages, _ = compile_prompt(langfuse_client, prompt_name, variables)
        except Exception as exc:
            error(f"  Failed to compile prompt: {exc}")
            failed += 1
            continue

        # Call LLM
        try:
            result = call_llm(
                messages,
                provider=llm_provider,
                model=llm_model,
                client_kwargs=llm_client_kwargs,
            )
            content = result["content"]
            ttft_ms = result["ttft_ms"]
            e2e_ms = result["e2e_ms"]
            total_ttft_ms += ttft_ms
            total_e2e_ms += e2e_ms
            completed += 1
        except Exception as exc:
            error(f"  LLM call failed: {exc}")
            failed += 1
            continue

        # Parse response
        decision, tag, body = parse_assist_response(content)

        if decision == "show":
            shown += 1
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            previous_suggestions.append(body)
            suggestions_display.append((tag or "unknown", body))
            print(f"  {C.GREEN}show|{tag}{C.RESET} · {C.DIM}{body[:80]}{'...' if len(body) > 80 else ''}{C.RESET}")
        else:
            skipped += 1
            print(f"  {C.DIM}skip{C.RESET}")

        print(f"  {C.DIM}TTFT: {ttft_ms:.0f}ms | E2E: {e2e_ms:.0f}ms{C.RESET}")

        # Write to log file
        local_file.write(fmt_step_block(
            step_num=step_idx + 1,
            total_steps=len(steps),
            prev_utt_idx=prev_utt_idx,
            curr_utt_idx=curr_utt_idx,
            phase=phase,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            transcript=transcript_display,
            state_summary=state_summary,
            decision=decision,
            tag=tag,
            body=body,
            suggestions_display=fmt_suggestions_list(suggestions_display),
        ))
        local_file.flush()

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # 12. Summary
    avg_ttft = (total_ttft_ms / completed) if completed else 0.0
    avg_e2e = (total_e2e_ms / completed) if completed else 0.0

    local_file.write(fmt_footer(
        total_steps=len(steps),
        shown=shown,
        skipped=skipped,
        avg_ttft=avg_ttft,
        avg_e2e=avg_e2e,
        tag_counts=tag_counts,
    ))
    local_file.close()

    header("Experiment Complete")
    print(f"  Steps:     {C.CYAN}{len(steps)}{C.RESET}")
    print(f"  Shown:     {C.GREEN}{shown}{C.RESET} ({int(100 * shown / len(steps)) if steps else 0}%)")
    print(f"  Skipped:   {C.DIM}{skipped}{C.RESET} ({int(100 * skipped / len(steps)) if steps else 0}%)")
    print(f"  Failed:    {C.RED}{failed}{C.RESET}")
    if completed > 0:
        print(f"  Avg TTFT:  {C.CYAN}{avg_ttft:.0f}ms{C.RESET}")
        print(f"  Avg E2E:   {C.CYAN}{avg_e2e:.0f}ms{C.RESET}")
    if tag_counts:
        tag_str = ", ".join(f"{t}={c}" for t, c in sorted(tag_counts.items()))
        print(f"  Tags:      {C.CYAN}{tag_str}{C.RESET}")
    print(f"\n  Results:   {C.CYAN}{local_path}{C.RESET}")
```

**Step 2: Commit**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): add main experiment loop with accumulation"
```

---

### Task 9: End-to-end test run

**Files:**
- No files to modify — this is a manual verification task

**Step 1: Run the full script against conv_1 with the objection assist prompt**

```bash
cd /Users/mikkeldahl/callbuddy_service/scripts && python run_assist_experiment.py
```

Interactive inputs:
- Prompt: `elg_b2c_objection_assist`
- Model: `3` (Gemini 3 Flash Minimal Thinking)
- State file: select the `conv_1_incremental_v9_20260215_140623.state.json`
- Confirm: `Y`

**Step 2: Verify the output file**

Check `experiments/<run_name>.txt`:
- Header shows correct metadata
- Each step shows transcript, state summary, decision
- `show` steps have output text
- `previous_suggestions` accumulates correctly across steps
- Summary shows correct show/skip counts and tag distribution

**Step 3: Fix any issues found during the test run**

If the script fails or output looks wrong, fix and re-run.

**Step 4: Commit any fixes**

```bash
git add scripts/run_assist_experiment.py
git commit -m "fix(assist-experiment): fixes from end-to-end test run"
```

---

### Task 10: Final commit with all files

**Step 1: Verify clean state**

```bash
git status
git log --oneline -5
```

**Step 2: If any uncommitted changes remain, commit them**

```bash
git add scripts/run_assist_experiment.py
git commit -m "feat(assist-experiment): complete assist experiment runner"
```
