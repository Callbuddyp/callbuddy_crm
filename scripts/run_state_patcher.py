#!/usr/bin/env python3
"""
Interactive CLI tool for running state-update-patch experiments against
incremental Langfuse datasets.

For each dataset item (sorted by utterance_index), the script:
  1. Extracts the new transcript segment (utterances since last step)
  2. Compiles the Langfuse prompt with accumulated state + new segment
  3. Calls the LLM to generate a patch list
  4. Applies the patches to accumulate state
  5. Logs the run to Langfuse as a dataset experiment
  6. Saves accumulated state per step to a local JSON file

Usage:
    cd scripts
    python experiment_state_generator.py
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from services.llm_client import get_llm_client, _retry_on_rate_limit, _messages_to_contents  # noqa: E402
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
# TXT log formatting
# ---------------------------------------------------------------------------

def _format_patches_txt(patches: List[dict]) -> str:
    """Format a patch list with box-drawing characters for the TXT log."""
    if not patches:
        lines = [
            "    \u256d\u2500 Patches (0)",
            "    \u2502  (no patches)",
            "    \u2570\u2500",
        ]
        return "\n".join(lines)

    lines = [f"    \u256d\u2500 Patches ({len(patches)})"]
    for i, patch in enumerate(patches, 1):
        op = patch.get("op", "unknown")
        lines.append(f"    \u2502  {i}. {op}")
        for key, value in patch.items():
            if key == "op":
                continue
            lines.append(f"    \u2502     {key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("    \u2570\u2500")
    return "\n".join(lines)


def _format_tool_calls_txt(tool_calls: List[dict]) -> str:
    """Format tool calls with box-drawing characters for the TXT log."""
    if not tool_calls:
        return ""
    lines: List[str] = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        args = tc.get("args", {})
        resp_len = tc.get("response_length", 0)
        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
        lines.append("    \u256d\u2500 Tool Call")
        lines.append(f"    \u2502  {name}({args_str})")
        lines.append(f"    \u2502  Response: {resp_len} chars")
        lines.append("    \u2570\u2500")
    return "\n".join(lines)


def _format_transcript_txt(
    utterances: List[Any],
    start_idx: int,
    end_idx: int,
    labels: Dict[int, str],
) -> str:
    """Format utterances for the TXT log with speaker labels and indented text."""
    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, buffer
        if current_speaker is None or not buffer:
            return
        label = labels.get(current_speaker, f"Speaker {current_speaker}")
        text = " ".join(buffer)
        parts.append(f"{label}:\n  {text}")
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


def write_txt_header(
    f,
    dataset_name: str,
    prompt_name: str,
    prompt_version: int,
    model_label: str,
    model_id: str,
    audio_name: str,
    run_name: str,
) -> None:
    """Write the header block to the TXT log file."""
    rule = "=" * 80
    f.write(f"{rule}\n")
    f.write(f"  STATE PATCHER EXPERIMENT LOG\n")
    f.write(f"{rule}\n")
    f.write(f"  Dataset:      {dataset_name}\n")
    f.write(f"  Prompt:       {prompt_name} (v{prompt_version})\n")
    f.write(f"  Model:        {model_label} ({model_id})\n")
    f.write(f"  Conversation: {audio_name}\n")
    f.write(f"  Run:          {run_name}\n")
    f.write(f"{rule}\n")


def write_txt_step(
    f,
    step: int,
    new_start: int,
    new_end: int,
    ttft_ms: float,
    e2e_ms: float,
    utterances: List[Any],
    labels: Dict[int, str],
    patches: List[dict],
    tool_calls: Optional[List[dict]] = None,
) -> None:
    """Write one annotated transcript segment + patches to the TXT log."""
    f.write(f"\n\n[Step {step} \u00b7 Utterances {new_start}\u2013{new_end}"
            f" \u00b7 {ttft_ms:.0f}ms TTFT \u00b7 {e2e_ms:.0f}ms E2E]\n\n")
    transcript = _format_transcript_txt(utterances, new_start, new_end, labels)
    f.write(transcript + "\n\n")
    if tool_calls:
        f.write(_format_tool_calls_txt(tool_calls) + "\n\n")
    f.write(_format_patches_txt(patches) + "\n")


def write_txt_footer(f, step_count: int) -> None:
    """Write closing line to the TXT log."""
    rule = "=" * 80
    f.write(f"\n\n{rule}\n")
    f.write(f"  {step_count} steps processed\n")
    f.write(f"{rule}\n")


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


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def deep_copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(state)


def upsert_by(array: List[dict], match_field: str, new_item: dict) -> None:
    for i, existing in enumerate(array):
        if existing.get(match_field) == new_item.get(match_field):
            array[i] = new_item
            return
    array.append(new_item)


def _phase_notes_key(state: Dict[str, Any]) -> str:
    """Return the state key for phase notes (supports both v8 and v9 schemas)."""
    if "current_phase_notes" in state:
        return "current_phase_notes"
    return "phase_progress"


def apply_patches(
    state: Dict[str, Any],
    patch_list: List[dict],
    phase_checkpoints: Optional[Dict[str, List[str]]] = None,
    provider_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """Apply patch operations to state. Supports both v7 and v8 schemas."""
    state = deep_copy_state(state)
    for patch in patch_list:
        op = patch.get("op")
        if op == "set_phase":
            new_phase = patch["phase"]
            # v8 new format: append phase entry with null checkpoints to phase_log
            if "phase_log" in state:
                entry: Dict[str, Any] = {"phase": new_phase}
                if phase_checkpoints and new_phase in phase_checkpoints:
                    for ck in phase_checkpoints[new_phase]:
                        entry[ck] = None
                state["phase_log"].append(entry)
            # v8/v9 format: archive current phase notes into phase_history
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
            state["phase"] = new_phase
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
        elif op == "upsert_customer_fact":
            item = {k: v for k, v in patch.items() if k != "op"}
            upsert_by(state["customer_facts"], "key", item)
        elif op == "upsert_pain_point":
            item = {k: v for k, v in patch.items() if k != "op"}
            upsert_by(state["pain_points"], "id", item)
        elif op == "upsert_value_prop":
            item = {k: v for k, v in patch.items() if k != "op"}
            upsert_by(state["value_props_delivered"], "id", item)
        elif op == "add_objection":
            item = {k: v for k, v in patch.items() if k != "op"}
            upsert_by(state["objections"], "id", item)
        elif op == "update_objection":
            obj_id = patch["id"]
            for obj in state["objections"]:
                if obj["id"] == obj_id:
                    obj["status"] = patch["status"]
                    if patch.get("resolution"):
                        obj["resolution"] = patch["resolution"]
                    break
            else:
                warn(f"update_objection: no objection with id='{obj_id}'")
        elif op == "add_sentiment_shift":
            # v7 compat
            if "sentiment_log" in state:
                item = {k: v for k, v in patch.items() if k != "op"}
                state["sentiment_log"].append(item)
        elif op == "add_commitment":
            item = {k: v for k, v in patch.items() if k != "op"}
            state["commitments"].append(item)
        elif op == "add_observation":
            # v7 compat
            if "observations" in state:
                state["observations"].append({"insight": patch["insight"]})
                if len(state["observations"]) > 10:
                    state["observations"] = state["observations"][-10:]
        elif op == "update_customer_profile":
            state["customer_profile"] = {
                k: v for k, v in patch.items() if k != "op"
            }
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
        else:
            warn(f"Unknown patch op: {op}")
    return state


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
        one_of.pop(provider_op_idx)
    else:
        one_of[provider_op_idx]["properties"]["provider_name"]["enum"] = competitor_ids

    return schema


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_speaker_map(base_dir: Path) -> dict:
    path = base_dir / "speaker_map.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        warn(f"Could not load speaker map: {exc}")
        return {}


def speaker_label_map(audio_stem: str, speaker_map: dict) -> Dict[int, str]:
    speaker1_is_seller = speaker_map.get(audio_stem)
    if not isinstance(speaker1_is_seller, bool):
        speaker1_is_seller = True
    return {
        1: "Sælger" if speaker1_is_seller else "Kunden",
        2: "Kunden" if speaker1_is_seller else "Sælger",
    }


def load_conversation(base_dir: Path, audio_stem: str) -> Tuple[str, Conversation]:
    """Load a processed conversation by audio_stem, returns (audio_name, conversation)."""
    conv_dir = base_dir / "processed_conversations"
    # Try exact match first, then glob
    candidates = list(conv_dir.glob(f"{audio_stem}.json")) + list(conv_dir.glob(f"{audio_stem}.*.json"))
    if not candidates:
        # Try all files and match by audio_stem inside
        for f in sorted(conv_dir.glob("*.json")):
            with f.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if raw.get("audio_stem") == audio_stem:
                conv = Conversation.from_dict(raw.get("conversation", {}))
                return raw.get("audio_name", audio_stem), conv
        raise FileNotFoundError(f"No processed conversation found for audio_stem={audio_stem} in {conv_dir}")

    path = candidates[0]
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    conv = Conversation.from_dict(raw.get("conversation", {}))
    return raw.get("audio_name", audio_stem), conv


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 5  # number of context utterances to include before the new segment


def _format_utterances(
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


def build_transcript_variables(
    utterances: List[Utterance],
    new_start_idx: int,
    new_end_idx: int,
    labels: Dict[int, str],
) -> Tuple[str, str]:
    """Build old_transcript (context) and new_transcript from utterance indices.

    Returns (old_transcript, new_transcript).
    old_transcript contains up to CONTEXT_WINDOW utterances before new_start_idx.
    new_transcript contains utterances[new_start_idx:new_end_idx+1].
    """
    # Context window: up to CONTEXT_WINDOW utterances before the new segment
    ctx_start = max(0, new_start_idx - CONTEXT_WINDOW)
    ctx_end = new_start_idx - 1  # inclusive

    old_transcript = ""
    if ctx_start <= ctx_end:
        old_transcript = _format_utterances(utterances, ctx_start, ctx_end, labels)

    new_transcript = _format_utterances(utterances, new_start_idx, new_end_idx, labels)
    return old_transcript, new_transcript


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def parse_patch_response(raw_text: str) -> List[dict]:
    """Strip code fences / thinking tags, parse JSON, extract patches list."""
    text = raw_text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    # Strip <think>...</think> blocks from thinking models
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # If text contains non-JSON before the actual object, extract the JSON portion
    if text and text[0] not in ("{", "["):
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if match:
            text = match.group(1)

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Direct "patches" key
        if "patches" in parsed:
            return parsed["patches"]
        # Thinking models may nest: {"thinking": "...", "result": {"patches": [...]}}
        for val in parsed.values():
            if isinstance(val, list) and (not val or isinstance(val[0], dict)):
                return val
            if isinstance(val, dict) and "patches" in val:
                return val["patches"]
    raise ValueError(f"Unexpected patch response shape: {type(parsed)}")


AVAILABLE_MODELS = {
    "1": {"provider": "groq", "model": "openai/gpt-oss-120b", "label": "GPT-OSS 120B (Groq)"},
    "2": {"provider": "groq", "model": "moonshotai/kimi-k2-instruct-0905", "label": "Kimi K2 Instruct (Groq)", "client_kwargs": {"reasoning_effort": None}},
    "3": {"provider": "gemini", "model": "gemini-3-flash-preview", "label": "Gemini 3 Flash"},
    "4": {"provider": "fireworks", "model": "accounts/fireworks/models/kimi-k2-thinking", "label": "Kimi K2 Thinking (Fireworks)"},
    "5": {"provider": "vertex", "model": "publishers/moonshotai/models/kimi-k2-thinking-maas", "label": "Kimi K2 Thinking (Vertex AI)"},
    "6": {"provider": "gemini-lite", "model": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Budget 1024 Thinking)"},
    "7": {"provider": "fireworks-lite", "model": "accounts/fireworks/models/kimi-k2-thinking", "label": "Kimi K2 Thinking (Fireworks, Budget 1024)"},
    "8": {"provider": "fireworks", "model": "accounts/fireworks/models/glm-4p7-flash", "label": "GLM-4p7 Flash (Fireworks)"},
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
        from langfuse.openai import OpenAI as LangfuseOpenAI

        api_key = load_env_value("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required")

        fw_client = LangfuseOpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")

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
            name="state_patcher",
            metadata={"provider": "fireworks"},
        )
        if response_format is not None:
            if response_format.get("type") == "json_schema":
                sanitized_rf = copy.deepcopy(response_format)
                inner = sanitized_rf.get("json_schema", {})
                if "schema" in inner:
                    from services.llm_client import _sanitize_schema_for_fireworks
                    inner["schema"] = _sanitize_schema_for_fireworks(inner["schema"])
                kwargs["response_format"] = sanitized_rf
            else:
                kwargs["response_format"] = response_format

        t0 = time.perf_counter()
        stream = fw_client.chat.completions.create(**kwargs)
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
            raise ValueError("Fireworks returned no content")

        return {"content": content, "ttft_ms": ttft_ms or e2e_ms, "e2e_ms": e2e_ms}

    # Non-streaming fallback for other providers
    client = get_llm_client(provider, **(client_kwargs or {}))
    t0 = time.perf_counter()
    content = client.generate(messages=messages, model=model, response_format=response_format)
    e2e_ms = (time.perf_counter() - t0) * 1000
    return {"content": content, "ttft_ms": e2e_ms, "e2e_ms": e2e_ms}


# ---------------------------------------------------------------------------
# Tool-calling support
# ---------------------------------------------------------------------------

_TOOL_SUPPORTED_PROVIDERS = ("gemini", "gemini-lite", "vertex", "fireworks", "fireworks-lite")



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
    from google import genai
    from google.genai import types

    if provider == "vertex":
        from services.otel_setup import setup_otel_tracing
        setup_otel_tracing()
        project = load_env_value("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        client = genai.Client(vertexai=True, project=project, location="global")
    else:
        api_key = load_env_value("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        client = genai.Client(api_key=api_key)
    # Convert OpenAI-style messages to types.Content objects (not plain dicts)
    # so they're compatible with types.Content objects appended during tool-calling
    raw_contents = _messages_to_contents(messages)
    if not raw_contents:
        raise ValueError("At least one message with text content is required")
    contents = [
        types.Content(role=c["role"], parts=[types.Part(text=p["text"]) for p in c["parts"]])
        for c in raw_contents
    ]

    # Build response schema kwargs (applied only after tools are removed)
    response_schema_kwargs: dict = {}
    if response_format is not None:
        fmt_type = response_format.get("type")
        if fmt_type in ("json_object", "json_schema"):
            response_schema_kwargs["response_mime_type"] = "application/json"
        if fmt_type == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                response_schema_kwargs["response_schema"] = schema

    # Build config with tools (no response schema — conflicts with tool calling)
    config_kwargs: dict = {
        "tools": [types.Tool(function_declarations=[tool_declaration])],
        "tool_config": types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO"),
        ),
        "thinking_config": types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW,
        ),
    }

    tool_calls_log: List[dict] = []
    max_turns = 3

    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None

    for turn in range(max_turns):
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000

        # Check for function call in response
        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
            raise ValueError("LLM returned empty response during tool-calling loop")

        function_call_part = None
        text_part = None
        for part in candidate.content.parts:
            if getattr(part, "function_call", None):
                function_call_part = part
            if getattr(part, "text", None):
                text_part = part

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
            # Remove tools, add response schema — force structured text response next
            config_kwargs.pop("tools", None)
            config_kwargs.pop("tool_config", None)
            config_kwargs.update(response_schema_kwargs)
            continue

        # Text response — we're done
        if text_part:
            e2e_ms = (time.perf_counter() - t0) * 1000
            return {
                "content": text_part.text,
                "ttft_ms": ttft_ms or e2e_ms,
                "e2e_ms": e2e_ms,
                "tool_calls": tool_calls_log,
            }

        raise ValueError("Gemini response contained neither function call nor text")

    # Exhausted max turns — return whatever we have
    e2e_ms = (time.perf_counter() - t0) * 1000
    raise ValueError(f"Tool-calling loop exhausted {max_turns} turns without text response")


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
    import json
    from langfuse.openai import OpenAI as LangfuseOpenAI

    api_key = load_env_value("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is required")

    fw_client = LangfuseOpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")

    merged_kw = client_kwargs or {}
    conversation: list = list(messages)

    tool_calls_log: List[dict] = []
    max_turns = 3

    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None

    tools = [tool_definition]

    for turn in range(max_turns):
        kwargs: dict = dict(
            model=model,
            messages=conversation,
            temperature=merged_kw.get("temperature", 0.6),
            max_tokens=merged_kw.get("max_tokens", 4000),
            top_p=merged_kw.get("top_p", 1),
            tools=tools,
            tool_choice="auto",
            # Langfuse trace metadata (picked up by the drop-in wrapper)
            name=f"state_patcher_tool_turn_{turn}",
            metadata={"provider": "fireworks", "turn": turn},
        )

        response = fw_client.chat.completions.create(**kwargs)
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000

        choice = response.choices[0]
        message = choice.message

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
            # Remove tools for next turn — force text response
            tools = []
            continue

        # Text response — we're done
        if message.content:
            e2e_ms = (time.perf_counter() - t0) * 1000
            return {
                "content": message.content,
                "ttft_ms": ttft_ms or e2e_ms,
                "e2e_ms": e2e_ms,
                "tool_calls": tool_calls_log,
            }

        raise ValueError("LLM response contained neither tool call nor text")

    e2e_ms = (time.perf_counter() - t0) * 1000
    raise ValueError(f"Tool-calling loop exhausted {max_turns} turns without text response")


# ---------------------------------------------------------------------------
# Langfuse integration
# ---------------------------------------------------------------------------

def fetch_dataset_items_sorted(langfuse_client, dataset_name: str) -> list:
    """Fetch all items from a dataset, sorted by utterance_index ascending."""
    dataset = _with_rate_limit_backoff(
        lambda: langfuse_client.get_dataset(name=dataset_name)
    )
    items = dataset.items
    items.sort(key=lambda item: item.metadata.get("utterance_index", 0))
    return items


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


def load_campaign_context(base_dir: Path) -> str:
    """Load campaign context from campaign_info/ directory if it exists."""
    info_dir = base_dir / "campaign_info"
    if not info_dir.exists():
        return ""
    parts = []
    for f in sorted(info_dir.iterdir()):
        if f.is_file() and f.suffix in (".txt", ".md"):
            parts.append(f.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Local conversation helpers (for all-conversations mode)
# ---------------------------------------------------------------------------

def get_customer_speaker_id(audio_stem: str, speaker_map: dict) -> int:
    """Return the speaker_id for the customer."""
    speaker1_is_seller = speaker_map.get(audio_stem)
    if not isinstance(speaker1_is_seller, bool):
        speaker1_is_seller = True
    return 2 if speaker1_is_seller else 1


def compute_customer_utterance_indices(
    utterances: List[Utterance],
    customer_speaker_id: int,
    min_time_ms: int = 0,
) -> List[int]:
    """Compute customer utterance indices, optionally filtered by min time gap."""
    indices = [i for i, u in enumerate(utterances) if u.speaker_id == customer_speaker_id]
    if min_time_ms > 0 and len(indices) > 1:
        filtered = [indices[0]]
        for idx in indices[1:]:
            prev_end = utterances[filtered[-1]].end_ms
            curr_start = utterances[idx].start_ms
            if (curr_start - prev_end) >= min_time_ms:
                filtered.append(idx)
        indices = filtered
    return indices


def load_all_processed_conversations(base_dir: Path) -> List[Tuple[str, str, Conversation]]:
    """Load all processed conversations from a campaign directory.

    Returns list of (audio_name, audio_stem, conversation).
    """
    conv_dir = base_dir / "processed_conversations"
    if not conv_dir.exists():
        return []
    results = []
    for f in sorted(conv_dir.glob("*.json")):
        with f.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        audio_name = raw.get("audio_name", f.stem)
        audio_stem = raw.get("audio_stem", audio_name.rsplit(".", 1)[0] if "." in audio_name else audio_name)
        conv = Conversation.from_dict(raw.get("conversation", {}))
        results.append((audio_name, audio_stem, conv))
    return results


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

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
    provider_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> List[dict]:
    """Run state patcher experiment over utterance indices.

    Returns the list of state steps for .state.json output.
    """
    # Prepare output files
    output_dir.mkdir(parents=True, exist_ok=True)
    state_json_path = output_dir / f"{run_name}.state.json"
    txt_path = output_dir / f"{run_name}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        write_txt_header(f, run_name, prompt_name, prompt_version, model_label, llm_model, audio_name, run_name)
    success(f"State JSON:  {state_json_path}")
    success(f"TXT log:     {txt_path}")

    # Main loop
    state = deep_copy_state(initial_state)
    prev_utterance_index = -1
    step_count = 0
    state_steps: List[dict] = []

    for step, utterance_index in enumerate(utterance_indices):
        # Derive old (context) and new transcript segments
        new_start = prev_utterance_index + 1
        new_end = utterance_index
        old_transcript, new_transcript = build_transcript_variables(
            utterances, new_start, new_end, labels
        )

        if not new_transcript.strip():
            warn(f"Step {step}: empty transcript segment (idx {new_start}-{new_end}), skipping")
            prev_utterance_index = utterance_index
            continue

        info(f"Step {step}: utterance_index={utterance_index} (new: {new_start}-{new_end})")
        print(f"  {C.DIM}{new_transcript[:120]}{'...' if len(new_transcript) > 120 else ''}{C.RESET}")

        # Compile prompt
        variables = {
            "current_state": json.dumps(state, ensure_ascii=False, indent=2),
            "old_transcript": old_transcript,
            "new_transcript": new_transcript,
        }
        if campaign_context:
            variables["campaign_context"] = campaign_context

        try:
            messages, _ = compile_prompt(langfuse_client, prompt_name, variables)
        except Exception as exc:
            error(f"Failed to compile prompt: {exc}")
            prev_utterance_index = utterance_index
            continue

        # Run LLM call
        raw_response = None
        patches = None

        try:
            llm_result = call_llm(messages, provider=llm_provider, model=llm_model, client_kwargs=llm_client_kwargs, response_format=response_format)

            raw_response = llm_result["content"]
            ttft_ms = llm_result["ttft_ms"]
            e2e_ms = llm_result["e2e_ms"]

            info(f"LLM responded — TTFT: {ttft_ms:.0f}ms | E2E: {e2e_ms:.0f}ms")

            # Parse and apply patches
            patches = parse_patch_response(raw_response)
            state = apply_patches(state, patches, phase_checkpoints=phase_checkpoints, provider_lookup=provider_lookup)

        except json.JSONDecodeError as exc:
            error(f"Failed to parse patch response: {exc}")
            if raw_response:
                print(f"  {C.DIM}Raw: {raw_response[:300]}{C.RESET}")
            prev_utterance_index = utterance_index
            continue
        except Exception as exc:
            error(f"Step {step} failed: {exc}")
            if raw_response:
                print(f"  {C.DIM}Raw: {raw_response[:300]}{C.RESET}")
            prev_utterance_index = utterance_index
            continue

        success(f"Applied {len(patches)} patches")
        print(f"\n{C.BOLD}Current state:{C.RESET}")
        print(f"{C.DIM}{json.dumps(state, ensure_ascii=False, indent=2)}{C.RESET}\n")

        # Collect state step
        state_steps.append({
            "utterance_index": utterance_index,
            "accumulated_state": deep_copy_state(state),
        })
        with txt_path.open("a", encoding="utf-8") as f:
            write_txt_step(f, step, new_start, new_end, ttft_ms, e2e_ms, utterances, labels, patches)
        step_count += 1

        prev_utterance_index = utterance_index
        progress(step + 1, len(utterance_indices), f"utt={utterance_index}")

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Write TXT footer
    with txt_path.open("a", encoding="utf-8") as f:
        write_txt_footer(f, step_count)

    # Write .state.json
    state_json_data = {
        "conversation": audio_name,
        "initial_state": initial_state,
        "steps": state_steps,
    }
    with state_json_path.open("w", encoding="utf-8") as f:
        json.dump(state_json_data, f, ensure_ascii=False, indent=2)

    success(f"Processed {step_count} steps for '{audio_name}'")
    return state_steps


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    header("State Experiment Generator")

    # 1. Prompt name
    prompt_name = prompt_string("Prompt name (from Langfuse)", default="state_update_prompt")

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

    # 3. Campaign path
    campaign_path = prompt_string(
        "Campaign path (relative to customer_data/)",
        prefix="customer_data/",
    )
    base_dir = (_PROJECT_ROOT / "customer_data" / campaign_path).resolve()
    if not base_dir.exists():
        error(f"Directory not found: {base_dir}")
        sys.exit(1)
    success(f"Using: {base_dir}")

    # 4. Mode selection
    print(f"\n{C.BOLD}Processing mode:{C.RESET}")
    print(f"  {C.CYAN}1{C.RESET}) Single dataset (from Langfuse)")
    print(f"  {C.CYAN}2{C.RESET}) Single conversation (from processed_conversations/)")
    print(f"  {C.CYAN}3{C.RESET}) All conversations (from processed_conversations/)")
    mode_choice = prompt_string("Select mode", default="2")
    if mode_choice not in ("1", "2", "3"):
        error(f"Invalid choice '{mode_choice}'. Using default.")
        mode_choice = "2"

    # Init Langfuse (needed for prompt compilation in both modes)
    langfuse_client = init_langfuse(push_to_langfuse=True)
    if langfuse_client is None:
        error("Could not initialize Langfuse. Check your .env credentials.")
        sys.exit(1)
    success("Langfuse initialized")

    # Ensure Langfuse env vars are set for the OpenAI drop-in wrapper
    for key in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"):
        val = load_env_value(key)
        if val:
            os.environ.setdefault(key, val)

    # Fetch prompt config (json_schema + initial_state)
    info(f"Fetching prompt config for '{prompt_name}'...")
    prompt_obj = _with_rate_limit_backoff(
        lambda: langfuse_client.get_prompt(name=prompt_name, type="chat")
    )
    prompt_version = prompt_obj.version
    prompt_config = prompt_obj.config or {}

    json_schema = prompt_config.get("json_schema")
    if not json_schema:
        error("Prompt config missing 'json_schema'.")
        sys.exit(1)

    initial_state = prompt_config.get("initial_state")
    if not initial_state:
        error("Prompt config missing 'initial_state'.")
        sys.exit(1)

    phase_checkpoints = prompt_config.get("phase_checkpoints")
    if phase_checkpoints:
        info(f"Loaded phase_checkpoints for {len(phase_checkpoints)} phases")

    success("Loaded json_schema and initial_state from prompt config")

    # Load campaign context
    campaign_context = load_campaign_context(base_dir)
    if campaign_context:
        info(f"Loaded campaign context ({len(campaign_context)} chars)")

    output_dir = base_dir / "state_experiments"
    speaker_map = load_speaker_map(base_dir)

    # Provider knowledge — load competitor IDs from Supabase for set_provider_knowledge
    competitor_ids: List[str] = []
    provider_lookup: Optional[Callable[[str], Optional[str]]] = None
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
                    def _lookup_provider(provider_name: str) -> Optional[str]:
                        return sb.get_competitor_description(campaign_id_input, provider_name)
                    provider_lookup = _lookup_provider
                else:
                    warn("No competitors found for this campaign in Supabase")
            else:
                warn("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY — skipping competitor loading")
        except Exception as exc:
            warn(f"Failed to load competitors from Supabase: {exc}")

    # Patch schema with real competitor enum (or remove op if no competitors)
    json_schema = patch_schema_competitor_enum(json_schema, competitor_ids)
    if competitor_ids:
        info(f"Patched schema: set_provider_knowledge enum = {competitor_ids}")
    else:
        info("Patched schema: removed set_provider_knowledge (no competitors)")

    # Rebuild response_format with patched schema
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "state_patches",
            "schema": json_schema,
            "strict": True,
        },
    }

    if mode_choice == "1":
        # --- Single dataset mode ---
        dataset_name = prompt_string("Dataset name", default="conv_1_incremental")

        default_run = f"{dataset_name}_v{prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_name = prompt_string("Run name", default=default_run)

        # Summary + confirm
        header("Experiment Summary")
        print(f"  Mode:       {C.CYAN}Single dataset{C.RESET}")
        print(f"  Dataset:    {C.CYAN}{dataset_name}{C.RESET}")
        print(f"  Prompt:     {C.CYAN}{prompt_name} (v{prompt_version}){C.RESET}")
        print(f"  Model:      {C.CYAN}{model_label} ({llm_model}){C.RESET}")
        print(f"  Campaign:   {base_dir}")
        print(f"  Run name:   {C.CYAN}{run_name}{C.RESET}")
        print()

        if not prompt_confirm("Proceed?"):
            info("Aborted.")
            sys.exit(0)

        # Fetch dataset items
        info(f"Fetching dataset '{dataset_name}'...")
        items = fetch_dataset_items_sorted(langfuse_client, dataset_name)
        if not items:
            error("No items found in dataset.")
            sys.exit(1)
        success(f"Found {len(items)} items (sorted by utterance_index)")

        # Extract utterance indices from dataset items
        utterance_indices = [
            (item.metadata or {}).get("utterance_index", 0) for item in items
        ]

        # Detect audio_stem from first item metadata
        first_meta = items[0].metadata or {}
        audio_stem = first_meta.get("audio_stem", "")
        if not audio_stem:
            error("First dataset item has no 'audio_stem' in metadata.")
            sys.exit(1)
        info(f"Audio stem: {audio_stem}")

        # Load conversation
        labels = speaker_label_map(audio_stem, speaker_map)
        audio_name, conversation = load_conversation(base_dir, audio_stem)
        utterances = conversation.utterances
        success(f"Loaded conversation '{audio_name}' with {len(utterances)} utterances")

        header("Running Experiment")
        run_experiment(
            langfuse_client=langfuse_client,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_client_kwargs=llm_client_kwargs,
            response_format=response_format,
            initial_state=initial_state,
            utterance_indices=utterance_indices,
            utterances=utterances,
            labels=labels,
            audio_name=audio_name,
            campaign_context=campaign_context,
            output_dir=output_dir,
            run_name=run_name,
            model_label=model_label,
            phase_checkpoints=phase_checkpoints,
            provider_lookup=provider_lookup,
        )

    elif mode_choice == "2":
        # --- Single conversation mode (from files) ---
        convs = load_all_processed_conversations(base_dir)
        if not convs:
            error("No processed conversations found.")
            sys.exit(1)

        print(f"\n{C.BOLD}Available conversations:{C.RESET}")
        for i, (audio_name, audio_stem, conv) in enumerate(convs, 1):
            print(f"  {C.CYAN}{i}{C.RESET}) {audio_stem} ({len(conv.utterances)} utterances)")
        conv_choice = prompt_string("Select conversation", default="1")
        try:
            conv_idx = int(conv_choice) - 1
            if not (0 <= conv_idx < len(convs)):
                raise ValueError
        except ValueError:
            error(f"Invalid choice '{conv_choice}'.")
            sys.exit(1)

        audio_name, audio_stem, conv = convs[conv_idx]
        labels = speaker_label_map(audio_stem, speaker_map)
        cid = get_customer_speaker_id(audio_stem, speaker_map)
        utterance_indices = compute_customer_utterance_indices(conv.utterances, cid)

        run_name = f"{audio_stem}_v{prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Summary + confirm
        header("Experiment Summary")
        print(f"  Mode:         {C.CYAN}Single conversation{C.RESET}")
        print(f"  Conversation: {C.CYAN}{audio_stem}{C.RESET} ({len(conv.utterances)} utterances, {len(utterance_indices)} steps)")
        print(f"  Prompt:       {C.CYAN}{prompt_name} (v{prompt_version}){C.RESET}")
        print(f"  Model:        {C.CYAN}{model_label} ({llm_model}){C.RESET}")
        print(f"  Campaign:     {base_dir}")
        print(f"  Run name:     {C.CYAN}{run_name}{C.RESET}")
        print()

        if not prompt_confirm("Proceed?"):
            info("Aborted.")
            sys.exit(0)

        header("Running Experiment")
        run_experiment(
            langfuse_client=langfuse_client,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_client_kwargs=llm_client_kwargs,
            response_format=response_format,
            initial_state=initial_state,
            utterance_indices=utterance_indices,
            utterances=conv.utterances,
            labels=labels,
            audio_name=audio_name,
            campaign_context=campaign_context,
            output_dir=output_dir,
            run_name=run_name,
            model_label=model_label,
            phase_checkpoints=phase_checkpoints,
            provider_lookup=provider_lookup,
        )

    else:
        # --- All conversations mode ---
        convs = load_all_processed_conversations(base_dir)
        if not convs:
            error("No processed conversations found.")
            sys.exit(1)
        info(f"Found {len(convs)} processed conversation(s)")

        min_time_ms = 0
        min_time_input = prompt_string("Min time between customer utterances (ms)", default="0")
        try:
            min_time_ms = int(min_time_input)
        except ValueError:
            warn("Invalid value, using 0")

        # Summary + confirm
        header("Experiment Summary")
        print(f"  Mode:          {C.CYAN}All conversations{C.RESET}")
        print(f"  Prompt:        {C.CYAN}{prompt_name} (v{prompt_version}){C.RESET}")
        print(f"  Model:         {C.CYAN}{model_label} ({llm_model}){C.RESET}")
        print(f"  Campaign:      {base_dir}")
        print(f"  Conversations: {C.CYAN}{len(convs)}{C.RESET}")
        print(f"  min_time_ms:   {C.CYAN}{min_time_ms}{C.RESET}")
        print()
        for audio_name, audio_stem, conv in convs:
            cid = get_customer_speaker_id(audio_stem, speaker_map)
            indices = compute_customer_utterance_indices(conv.utterances, cid, min_time_ms)
            print(f"    {audio_stem}: {len(conv.utterances)} utterances, {len(indices)} steps")
        print()

        if not prompt_confirm("Proceed?"):
            info("Aborted.")
            sys.exit(0)

        for conv_idx, (audio_name, audio_stem, conv) in enumerate(convs):
            header(f"Conversation {conv_idx + 1}/{len(convs)}: {audio_stem}")

            labels = speaker_label_map(audio_stem, speaker_map)
            cid = get_customer_speaker_id(audio_stem, speaker_map)
            utterance_indices = compute_customer_utterance_indices(
                conv.utterances, cid, min_time_ms
            )

            if not utterance_indices:
                warn(f"No customer utterances found for '{audio_stem}', skipping")
                continue

            run_name = f"{audio_stem}_v{prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            info(f"Run name: {run_name} ({len(utterance_indices)} steps)")

            run_experiment(
                langfuse_client=langfuse_client,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_client_kwargs=llm_client_kwargs,
                response_format=response_format,
                initial_state=initial_state,
                utterance_indices=utterance_indices,
                utterances=conv.utterances,
                labels=labels,
                audio_name=audio_name,
                campaign_context=campaign_context,
                output_dir=output_dir,
                run_name=run_name,
                model_label=model_label,
                phase_checkpoints=phase_checkpoints,
                competitor_tool_service=competitor_tool_service,
                competitor_prompt_name=competitor_prompt_name,
            )

    # Flush Langfuse to ensure all traces are sent
    langfuse_client.flush()

    header("Done!")


if __name__ == "__main__":
    main()
