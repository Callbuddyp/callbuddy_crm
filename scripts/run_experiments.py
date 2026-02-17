#!/usr/bin/env python3
"""
Generic Langfuse experiment runner.

Takes any dataset + any prompt + any model, runs the LLM against each dataset
item, and logs results to Langfuse using the experiments SDK (item.run()
context manager).  This enables comparing different prompt/model combinations
on the same dataset in Langfuse's experiment UI.

Usage:
    cd scripts
    python run_experiments.py
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
from services.llm_client import get_llm_client, _retry_on_rate_limit  # noqa: E402
from utils import load_env_value  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI color helpers
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


def progress(current: int, total: int, label: str = "") -> None:
    width = 30
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    pct = int(100 * current / total) if total else 100
    end = "\n" if current == total else "\r"
    print(f"  [{bar}] {pct:3d}% ({current}/{total}) {label}", end=end, flush=True)


# ---------------------------------------------------------------------------
# Local result file formatting
# ---------------------------------------------------------------------------

_RULE = "=" * 80
_THIN_RULE = "\u2500" * 80  # ────


def fmt_file_header(
    dataset_name: str,
    prompt_name: str,
    prompt_version: int,
    model_label: str,
    llm_model: str,
    num_items: int,
    run_name: str,
) -> str:
    return (
        f"{_RULE}\n"
        f"  EXPERIMENT LOG\n"
        f"{_RULE}\n"
        f"  Dataset:      {dataset_name}\n"
        f"  Prompt:       {prompt_name} (v{prompt_version})\n"
        f"  Model:        {model_label} ({llm_model})\n"
        f"  Items:        {num_items}\n"
        f"  Run:          {run_name}\n"
        f"{_RULE}\n"
    )


def fmt_item_input(item_input: dict) -> str:
    parts: list[str] = []
    for key, value in item_input.items():
        if isinstance(value, (dict, list)):
            parts.append(f"{key}:\n  {json.dumps(value, ensure_ascii=False, indent=2)}")
        else:
            parts.append(f"{key}:\n  {value}")
    return "\n\n".join(parts)


def fmt_item_block(
    idx: int, total: int, ttft_ms: float, e2e_ms: float,
    item_input: dict, output: str,
) -> str:
    return (
        f"{_THIN_RULE}\n"
        f"[Item {idx}/{total} \u00b7 {ttft_ms:.0f}ms TTFT \u00b7 {e2e_ms:.0f}ms E2E]\n\n"
        f"\u2500\u2500 Input \u2500\u2500\n"
        f"{fmt_item_input(item_input)}\n\n"
        f"\u2500\u2500 Output \u2500\u2500\n"
        f"{output}\n"
        f"{_THIN_RULE}\n"
    )


def fmt_item_error(idx: int, total: int, item_input: dict, err: str) -> str:
    return (
        f"{_THIN_RULE}\n"
        f"[Item {idx}/{total} \u00b7 FAILED]\n\n"
        f"\u2500\u2500 Input \u2500\u2500\n"
        f"{fmt_item_input(item_input)}\n\n"
        f"\u2500\u2500 Error \u2500\u2500\n"
        f"{err}\n"
        f"{_THIN_RULE}\n"
    )


def fmt_footer(completed: int, failed: int, total: int, avg_ttft: float, avg_e2e: float) -> str:
    return (
        f"\n{_RULE}\n"
        f"  SUMMARY\n"
        f"{_RULE}\n"
        f"  Completed: {completed} / {total}\n"
        f"  Failed:    {failed}\n"
        f"  Avg TTFT:  {avg_ttft:.0f}ms\n"
        f"  Avg E2E:   {avg_e2e:.0f}ms\n"
        f"{_RULE}\n"
    )


# ---------------------------------------------------------------------------
# CLI input helpers
# ---------------------------------------------------------------------------

def prompt_string(question: str, default: str = "", prefix: str = "") -> str:
    suffix = f" [{prefix}{default}]" if default else ""
    raw = input(f"{C.BOLD}{question}{suffix}: {C.RESET}").strip()
    if not raw:
        return default
    return raw


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
DEFAULT_MODEL_KEY = "1"


@_retry_on_rate_limit
def call_llm(
    messages: List[Dict[str, str]],
    provider: str,
    model: str,
    client_kwargs: Optional[Dict] = None,
    response_format: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Call LLM and return content + timing metrics (TTFT / E2E).

    Returns {"content": str, "ttft_ms": float, "e2e_ms": float}.
    Uses streaming for Groq to measure real TTFT.
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
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}

        # --- timed section: only the streaming API call ---
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

    if provider == "fireworks":
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
        if response_format is not None:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        stream = fw_client.chat.completions.create(**kwargs)
        ttft_ms: Optional[float] = None
        content_ttft_ms: Optional[float] = None
        chunks: List[str] = []
        for chunk in stream:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            delta = chunk.choices[0].delta
            # Thinking models emit reasoning tokens before content tokens
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

    # Non-streaming fallback for other providers
    client = get_llm_client(provider, **(client_kwargs or {}))
    t0 = time.perf_counter()
    content = client.generate(messages=messages, model=model, response_format=response_format)
    e2e_ms = (time.perf_counter() - t0) * 1000
    return {"content": content, "ttft_ms": e2e_ms, "e2e_ms": e2e_ms}


# ---------------------------------------------------------------------------
# Langfuse prompt helpers
# ---------------------------------------------------------------------------

def compile_prompt(
    langfuse_client, prompt_name: str, variables: Dict[str, str],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Fetch a prompt from Langfuse, compile with variables, return (messages, config).

    Supports both chat prompts (returns list of message dicts) and
    text prompts (returns a string, wrapped as a single user message).
    The config dict comes from prompt.config (may be empty).
    """
    # Try chat type first; fall back to text if it fails or returns a string
    try:
        prompt = _with_rate_limit_backoff(
            lambda: langfuse_client.get_prompt(name=prompt_name, type="chat")
        )
        compiled = prompt.compile(**variables)
        if isinstance(compiled, list):
            return compiled, prompt.config or {}
    except Exception:
        pass

    # Text prompt fallback
    prompt = _with_rate_limit_backoff(
        lambda: langfuse_client.get_prompt(name=prompt_name, type="text")
    )
    compiled = prompt.compile(**variables)
    return [{"role": "user", "content": str(compiled)}], prompt.config or {}


# ---------------------------------------------------------------------------
# New helpers for experiment runner
# ---------------------------------------------------------------------------

def prepare_template_variables(item_input: dict) -> Dict[str, str]:
    """Convert dataset item.input to string-valued template variables.

    Dict/list values are JSON-serialized; everything else is str().
    """
    variables: Dict[str, str] = {}
    for key, value in item_input.items():
        if isinstance(value, (dict, list)):
            variables[key] = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            variables[key] = str(value)
    return variables


def resolve_response_format(prompt_config: dict) -> Optional[dict]:
    """Extract response_format from prompt config if present."""
    rf = prompt_config.get("response_format")
    if rf == "json_object" or rf == {"type": "json_object"}:
        return {"type": "json_object"}
    if isinstance(rf, dict) and rf.get("type"):
        return rf
    # Fall back: if config has a json_schema key, use json_object mode
    if prompt_config.get("json_schema"):
        return {"type": "json_object"}
    return None


def extract_template_variables(prompt_obj) -> set:
    """Extract all {{variable}} names from a Langfuse prompt object.

    Works for both chat prompts (list of message dicts) and text prompts (string).
    Scans the raw prompt content before compilation so composability tags are
    already resolved but {{variables}} are still present.
    """
    variables: set = set()
    raw = prompt_obj.prompt
    if isinstance(raw, list):
        # Chat prompt: list of message dicts
        for msg in raw:
            content = msg.get("content", "")
            if isinstance(content, str):
                variables.update(re.findall(r"\{\{(\w+)\}\}", content))
    elif isinstance(raw, str):
        variables.update(re.findall(r"\{\{(\w+)\}\}", raw))
    return variables


# ---------------------------------------------------------------------------
# Assist experiment helpers
# ---------------------------------------------------------------------------

ASSIST_TRANSCRIPT_WINDOW = 5


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


def format_utterances(
    utterances: List[Utterance],
    start_idx: int,
    end_idx: int,
    labels: Dict[int, str],
) -> str:
    """Format utterances[start_idx:end_idx+1] with speaker labels, merging consecutive same-speaker turns."""
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
    window: int = ASSIST_TRANSCRIPT_WINDOW,
) -> str:
    """Build a transcript window: the last `window` utterances up to end_idx (inclusive)."""
    start_idx = max(0, end_idx - window + 1)
    return format_utterances(utterances, start_idx, end_idx, labels)


def build_assist_template_variables(
    utterances: List[Utterance],
    current_utt_idx: int,
    state: dict,
    previous_suggestions: List[str],
    labels: Dict[int, str],
) -> Dict[str, str]:
    """Build the template variables dict for the assist prompt."""
    transcript = build_transcript_window(utterances, current_utt_idx, labels)
    state_str = json.dumps(state, ensure_ascii=False, indent=2)
    suggestions_str = "\n---\n".join(previous_suggestions) if previous_suggestions else "(none)"
    return {
        "new_transcript": transcript,
        "transcript": transcript,
        "state": state_str,
        "previous_suggestions": suggestions_str,
    }


def parse_assist_response(raw: str) -> Tuple[str, Optional[str], str]:
    """Parse an assist prompt response.

    Returns (decision, tag, body):
    - decision: "show" or "skip"
    - tag: the tag string (e.g. "objection") if show, else None
    - body: the response text after the first line (empty string for skip)
    """
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

    warn(f"Unexpected first line format: '{lines[0].strip()[:60]}' — treating as show|unknown")
    return "show", "unknown", text


def discover_state_files(search_dir: Path) -> List[Path]:
    """Find all .state.json files under search_dir recursively."""
    return sorted(search_dir.rglob("*.state.json"))


def infer_campaign_dir(state_path: Path) -> Path:
    """Infer the campaign base directory from a .state.json path."""
    candidate = state_path.parent.parent
    if (candidate / "processed_conversations").exists():
        return candidate
    candidate = state_path.parent
    if (candidate / "processed_conversations").exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find processed_conversations/ relative to {state_path}. "
        f"Expected structure: customer_data/<firm>/<campaign>/state_experiments/<file>.state.json"
    )


def fmt_assist_state_summary(state: dict) -> str:
    """Format a compact state summary for the assist log."""
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


def fmt_assist_suggestions_list(suggestions: List[Tuple[str, str]]) -> str:
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


def fmt_assist_file_header(
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


def fmt_assist_step_block(
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
        f"\u2500\u2500 Transcript (last {ASSIST_TRANSCRIPT_WINDOW} utterances) \u2500\u2500\n"
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


def fmt_assist_footer(
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


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def run_dataset_experiment() -> None:
    """Run a dataset-based experiment: prompt + model against Langfuse dataset items."""
    header("Dataset Experiment")

    # 1. Prompt name
    prompt_name = prompt_string("Prompt name (from Langfuse)")
    if not prompt_name:
        error("Prompt name is required.")
        sys.exit(1)

    # 2. Model selection
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

    # 3. Dataset name
    dataset_name = prompt_string("Dataset name (from Langfuse)")
    if not dataset_name:
        error("Dataset name is required.")
        sys.exit(1)

    # 4. Init Langfuse + fetch prompt
    langfuse_client = init_langfuse(push_to_langfuse=True)
    if langfuse_client is None:
        error("Could not initialize Langfuse. Check your .env credentials.")
        sys.exit(1)
    success("Langfuse initialized")

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
    prompt_config = prompt_obj.config or {}
    prompt_variables = extract_template_variables(prompt_obj)
    success(f"Prompt '{prompt_name}' v{prompt_version}")
    if prompt_variables:
        info(f"Prompt template variables: {', '.join(sorted(prompt_variables))}")

    # Resolve response_format from prompt config
    response_format = resolve_response_format(prompt_config)
    if response_format:
        info(f"Response format: {response_format.get('type', 'unknown')}")

    # 5. Run name
    # Build label-safe model name (e.g. "gemini-3-flash")
    model_slug = model_label.lower().replace(" ", "-").replace("(", "").replace(")", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run = f"{model_slug}_{prompt_name}_v{prompt_version}_{timestamp}"
    run_name = prompt_string("Run name", default=default_run)

    # 6. Fetch dataset
    info(f"Fetching dataset '{dataset_name}'...")
    try:
        dataset = _with_rate_limit_backoff(
            lambda: langfuse_client.get_dataset(name=dataset_name)
        )
    except Exception as exc:
        error(f"Failed to fetch dataset '{dataset_name}': {exc}")
        sys.exit(1)

    items = dataset.items
    if not items:
        error("No items found in dataset.")
        sys.exit(1)
    success(f"Found {len(items)} dataset items")

    # 6b. Compare prompt template variables with dataset item keys
    dataset_keys = set(items[0].input.keys()) if items[0].input else set()
    matched_vars = prompt_variables & dataset_keys
    missing_vars = prompt_variables - dataset_keys  # prompt expects but dataset lacks
    extra_keys = dataset_keys - prompt_variables     # dataset has but prompt ignores

    default_variables: Dict[str, str] = {}

    if missing_vars or extra_keys:
        header("Variable Mapping")
        if matched_vars:
            print(f"  {C.GREEN}Matched:{C.RESET}  {', '.join(sorted(matched_vars))}")
        if missing_vars:
            print(f"  {C.YELLOW}Missing:{C.RESET}  {', '.join(sorted(missing_vars))}")
            print(f"           (prompt expects these but dataset items don't have them)")
        if extra_keys:
            print(f"  {C.DIM}Unused:{C.RESET}   {', '.join(sorted(extra_keys))}")
            print(f"           (dataset items have these but prompt doesn't use them)")
        print()

        if missing_vars:
            warn("Missing variables will be sent as empty strings unless you provide defaults.")
            for var in sorted(missing_vars):
                default_val = prompt_string(f"Default for '{var}'", default="")
                default_variables[var] = default_val
            print()

    # 7. Summary + confirm
    header("Experiment Summary")
    print(f"  Prompt:       {C.CYAN}{prompt_name}{C.RESET} (v{prompt_version})")
    print(f"  Model:        {C.CYAN}{model_label} ({llm_model}){C.RESET}")
    print(f"  Dataset:      {C.CYAN}{dataset_name}{C.RESET} ({len(items)} items)")
    print(f"  Run name:     {C.CYAN}{run_name}{C.RESET}")
    if response_format:
        print(f"  Resp. format: {C.CYAN}{response_format.get('type', 'unknown')}{C.RESET}")
    if default_variables:
        print(f"  Defaults:     {C.YELLOW}{len(default_variables)} variable(s){C.RESET}")
        for k, v in default_variables.items():
            display = v[:60] + "..." if len(v) > 60 else v
            display = display if display else "(empty)"
            print(f"                  {k} = {C.DIM}{display}{C.RESET}")
    if extra_keys:
        print(f"  Unused keys:  {C.DIM}{', '.join(sorted(extra_keys))}{C.RESET}")
    print()

    if not prompt_confirm("Proceed?"):
        info("Aborted.")
        sys.exit(0)

    # 7b. Save results locally?
    save_local = prompt_confirm("Save results locally?")
    local_file = None
    if save_local:
        experiments_dir = _PROJECT_ROOT / "experiments" / prompt_name
        experiments_dir.mkdir(parents=True, exist_ok=True)
        local_path = experiments_dir / f"{run_name}.txt"
        local_file = open(local_path, "w", encoding="utf-8")
        local_file.write(fmt_file_header(
            dataset_name=dataset_name,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            model_label=model_label,
            llm_model=llm_model,
            num_items=len(items),
            run_name=run_name,
        ))
        local_file.flush()
        success(f"Saving results to {local_path}")

    # 8. Run experiment
    header("Running Experiment")

    completed = 0
    failed = 0
    total_ttft_ms = 0.0
    total_e2e_ms = 0.0

    for idx, item in enumerate(items):
        info(f"Item {idx + 1}/{len(items)}")

        try:
            variables = prepare_template_variables(item.input)
            variables.update(default_variables)  # fill in missing prompt variables
            messages, _ = compile_prompt(langfuse_client, prompt_name, variables)
        except Exception as exc:
            error(f"  Failed to compile prompt: {exc}")
            failed += 1
            if local_file:
                local_file.write(fmt_item_error(
                    idx=idx + 1, total=len(items),
                    item_input=item.input, err=f"Prompt compile: {exc}",
                ))
                local_file.flush()
            progress(idx + 1, len(items))
            continue

        try:
            with item.run(
                run_name=run_name,
                run_metadata={"model": llm_model, "provider": llm_provider},
            ) as root_span:
                try:
                    llm_result = call_llm(
                        messages,
                        provider=llm_provider,
                        model=llm_model,
                        client_kwargs=llm_client_kwargs,
                        response_format=response_format,
                    )
                    content = llm_result["content"]
                    ttft_ms = llm_result["ttft_ms"]
                    e2e_ms = llm_result["e2e_ms"]

                    root_span.update_trace(
                        input=item.input,
                        output=content,
                    )

                    total_ttft_ms += ttft_ms
                    total_e2e_ms += e2e_ms
                    completed += 1

                    print(f"  {C.DIM}TTFT: {ttft_ms:.0f}ms | E2E: {e2e_ms:.0f}ms | "
                          f"Output: {content[:80]}{'...' if len(content) > 80 else ''}{C.RESET}")

                    if local_file:
                        local_file.write(fmt_item_block(
                            idx=idx + 1, total=len(items),
                            ttft_ms=ttft_ms, e2e_ms=e2e_ms,
                            item_input=item.input, output=content,
                        ))
                        local_file.flush()

                except Exception as exc:
                    error(f"  LLM call failed: {exc}")
                    root_span.update_trace(
                        input=item.input,
                        output=None,
                        metadata={"error": str(exc)},
                    )
                    failed += 1
                    if local_file:
                        local_file.write(fmt_item_error(
                            idx=idx + 1, total=len(items),
                            item_input=item.input, err=str(exc),
                        ))
                        local_file.flush()

        except Exception as exc:
            error(f"  item.run() failed: {exc}")
            failed += 1
            if local_file:
                local_file.write(fmt_item_error(
                    idx=idx + 1, total=len(items),
                    item_input=item.input, err=f"item.run(): {exc}",
                ))
                local_file.flush()

        progress(idx + 1, len(items))

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Flush all pending data to Langfuse
    info("Flushing data to Langfuse...")
    langfuse_client.flush()

    # 9. Summary
    avg_ttft = (total_ttft_ms / completed) if completed else 0.0
    avg_e2e = (total_e2e_ms / completed) if completed else 0.0

    if local_file:
        local_file.write(fmt_footer(
            completed=completed, failed=failed,
            total=len(items), avg_ttft=avg_ttft, avg_e2e=avg_e2e,
        ))
        local_file.close()
        success(f"Results saved to {local_file.name}")

    header("Experiment Complete")
    print(f"  Completed:    {C.GREEN}{completed}{C.RESET}")
    print(f"  Failed:       {C.RED}{failed}{C.RESET}")
    if completed > 0:
        print(f"  Avg TTFT:     {avg_ttft:.0f}ms")
        print(f"  Avg E2E:      {avg_e2e:.0f}ms")
    print(f"  Run name:     {C.CYAN}{run_name}{C.RESET}")
    print()

    success("Done! Check Langfuse experiments UI for results.")


# ---------------------------------------------------------------------------
# Assist experiment mode
# ---------------------------------------------------------------------------

def run_assist_experiment() -> None:
    """Run an assist experiment: walk through a conversation with pre-computed state."""
    header("Assist Experiment")

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

    # 10. Open local file
    experiments_dir = _PROJECT_ROOT / "experiments" / prompt_name
    experiments_dir.mkdir(parents=True, exist_ok=True)
    local_path = experiments_dir / f"{run_name}.txt"
    local_file = open(local_path, "w", encoding="utf-8")
    local_file.write(fmt_assist_file_header(
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

    previous_suggestions: List[str] = []
    suggestions_display: List[Tuple[str, str]] = []
    total_ttft_ms = 0.0
    total_e2e_ms = 0.0
    completed = 0
    failed = 0
    shown = 0
    skipped = 0
    tag_counts: Dict[str, int] = {}

    for step_idx, step in enumerate(steps):
        curr_utt_idx = step["utterance_index"]
        if step_idx == 0:
            state = initial_state
        else:
            state = steps[step_idx - 1]["accumulated_state"]

        prev_utt_idx = steps[step_idx - 1]["utterance_index"] if step_idx > 0 else 0
        phase = state.get("phase", "?")

        info(f"Step {step_idx + 1}/{len(steps)} · Utterance {prev_utt_idx}→{curr_utt_idx} · Phase: {phase}")

        variables = build_assist_template_variables(
            utterances=utterances,
            current_utt_idx=curr_utt_idx,
            state=state,
            previous_suggestions=previous_suggestions,
            labels=labels,
        )

        transcript_display = variables["new_transcript"]
        state_summary = fmt_assist_state_summary(state)

        try:
            messages, _ = compile_prompt(langfuse_client, prompt_name, variables)
        except Exception as exc:
            error(f"  Failed to compile prompt: {exc}")
            failed += 1
            continue

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

        local_file.write(fmt_assist_step_block(
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
            suggestions_display=fmt_assist_suggestions_list(suggestions_display),
        ))
        local_file.flush()

        time.sleep(0.5)

    # 12. Summary
    avg_ttft = (total_ttft_ms / completed) if completed else 0.0
    avg_e2e = (total_e2e_ms / completed) if completed else 0.0

    local_file.write(fmt_assist_footer(
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    header("Langfuse Experiment Runner")

    mode_idx = prompt_choice("Select experiment mode", [
        "Dataset experiment (prompt + model vs Langfuse dataset)",
        "Assist experiment (walk conversation with pre-computed state)",
    ])

    if mode_idx == 0:
        run_dataset_experiment()
    else:
        run_assist_experiment()


if __name__ == "__main__":
    main()
