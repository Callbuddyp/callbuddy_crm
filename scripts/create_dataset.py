#!/usr/bin/env python3
"""
Interactive CLI tool for generating Langfuse test datasets from processed
conversation transcriptions.

Four dataset types:
  1. Incremental           – one test case per customer utterance (transcript up to that point)
  2. Full                  – one test case per conversation (complete transcript)
  3. Stateful Incremental  – per-step state + context window + new transcript segment,
                             aligned with state patcher step boundaries
  4. Evaluation Sampler    – LLM-selected quality-diagnostic subsample with 4 input vars
                             (transcript, state, old_transcript, new_transcript)

Usage:
    cd scripts
    python create_dataset.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
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

from models.conversation import Conversation, Utterance  # noqa: E402
from services.langfuse import init_langfuse, _with_rate_limit_backoff  # noqa: E402
from services.llm_client import get_llm_client  # noqa: E402

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
# CLI input helpers
# ---------------------------------------------------------------------------

def prompt_choice(question: str, options: List[str]) -> str:
    print(f"\n{C.BOLD}{question}{C.RESET}")
    for i, opt in enumerate(options, 1):
        print(f"  {C.CYAN}{i}{C.RESET}. {opt}")
    while True:
        raw = input(f"{C.DIM}> {C.RESET}").strip()
        if raw.lower() in [o.lower() for o in options]:
            return raw.lower()
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        warn("Invalid choice, try again.")


def prompt_int(question: str, default: Optional[int] = None) -> int:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{C.BOLD}{question}{suffix}: {C.RESET}").strip()
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            warn("Please enter a valid integer.")


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
# Data loading
# ---------------------------------------------------------------------------

def load_config(base_dir: Path) -> dict:
    path = base_dir / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"config.json not found in {base_dir}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def get_customer_speaker_id(audio_stem: str, speaker_map: dict) -> int:
    speaker1_is_seller = speaker_map.get(audio_stem)
    if not isinstance(speaker1_is_seller, bool):
        speaker1_is_seller = True
    return 2 if speaker1_is_seller else 1


def load_processed_conversations(base_dir: Path) -> List[Path]:
    conv_dir = base_dir / "processed_conversations"
    if not conv_dir.exists():
        raise FileNotFoundError(f"processed_conversations/ not found in {base_dir}")
    files = sorted(conv_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {conv_dir}")
    return files


def parse_conversation(raw_data: dict) -> Tuple[str, str, Conversation]:
    audio_name = raw_data.get("audio_name", "unknown")
    audio_stem = raw_data.get("audio_stem", audio_name.rsplit(".", 1)[0] if "." in audio_name else audio_name)
    conv_dict = raw_data.get("conversation", {})
    conversation = Conversation.from_dict(conv_dict)
    return audio_name, audio_stem, conversation


def load_experiment_state_map(path: str) -> Tuple[str, Dict[str, Any], List[Tuple[int, Dict[str, Any]]]]:
    """Load a .state.json file from the state patcher.

    Returns (audio_stem, initial_state, steps) where steps is a sorted list of
    (utterance_index, accumulated_state) tuples.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Experiment file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "steps" not in data or "conversation" not in data:
        raise ValueError("Experiment file must contain 'conversation' and 'steps' fields")

    # Extract audio_stem: strip file extension from the conversation field
    conversation = data["conversation"]
    audio_stem = conversation.rsplit(".", 1)[0] if "." in conversation else conversation

    initial_state = data.get("initial_state", {})

    steps: List[Tuple[int, Dict[str, Any]]] = []
    for step in data["steps"]:
        utt_idx = step.get("utterance_index")
        acc_state = step.get("accumulated_state", {})
        if utt_idx is not None:
            steps.append((int(utt_idx), acc_state))

    steps.sort(key=lambda s: s[0])
    return audio_stem, initial_state, steps


def load_experiment_state_dir(dir_path: str) -> List[Tuple[str, Dict[str, Any], List[Tuple[int, Dict[str, Any]]]]]:
    """Load all .state.json files from a directory.

    Returns list of (audio_stem, initial_state, steps) tuples.
    """
    p = Path(dir_path).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")
    results = []
    for f in sorted(p.glob("*.state.json")):
        try:
            audio_stem, initial_state, steps = load_experiment_state_map(str(f))
            results.append((audio_stem, initial_state, steps))
        except Exception as exc:
            warn(f"Skipping {f.name}: {exc}")
    return results


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------

def format_transcript_up_to(
    utterances: List[Utterance],
    end_index: int,
    labels: Dict[int, str],
) -> str:
    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, buffer
        if current_speaker is None or not buffer:
            return
        label = labels.get(current_speaker, f"Speaker {current_speaker}")
        parts.append(f"{label}: {' '.join(buffer)}")
        buffer = []

    for index, utt in enumerate(utterances):
        if index >= end_index:
            break
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
    return "\n".join(parts)


CONTEXT_WINDOW = 5  # number of context utterances before a new segment


def format_utterance_range(
    utterances: List[Utterance],
    start_idx: int,
    end_idx: int,
    labels: Dict[int, str],
) -> str:
    """Format utterances[start_idx..end_idx] (inclusive) with speaker labels."""
    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, buffer
        if current_speaker is None or not buffer:
            return
        label = labels.get(current_speaker, f"Speaker {current_speaker}")
        parts.append(f"{label}: {' '.join(buffer)}")
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
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def _find_next_seller_text(
    utterances: List[Utterance],
    after_index: int,
    customer_speaker_id: int,
) -> str:
    for i in range(after_index + 1, len(utterances)):
        if utterances[i].speaker_id != customer_speaker_id:
            return utterances[i].cleaned_text
    return ""


def generate_incremental_items(
    audio_name: str,
    audio_stem: str,
    conversation: Conversation,
    labels: Dict[int, str],
    min_time_ms: int,
    customer_speaker_id: int,
) -> List[dict]:
    utterances = conversation.utterances

    # Find all customer utterance indices
    customer_indices: List[int] = [
        i for i, u in enumerate(utterances) if u.speaker_id == customer_speaker_id
    ]

    # Filter by min_time_ms gap
    if min_time_ms > 0 and len(customer_indices) > 1:
        filtered = [customer_indices[0]]
        for idx in customer_indices[1:]:
            prev_end = utterances[filtered[-1]].end_ms
            curr_start = utterances[idx].start_ms
            if (curr_start - prev_end) >= min_time_ms:
                filtered.append(idx)
        customer_indices = filtered

    items: List[dict] = []
    for idx in customer_indices:
        # Transcript up to and including this customer utterance
        transcript = format_transcript_up_to(utterances, idx + 1, labels)
        expected = _find_next_seller_text(utterances, idx, customer_speaker_id)

        items.append({
            "input": {"transcript": transcript},
            "metadata": {
                "audio_stem": audio_stem,
                "audio_name": audio_name,
                "type": "incremental",
                "utterance_index": idx,
                "conversation": audio_name,
            },
            "expected_output": expected,
        })

    # Reverse order (last utterance first)
    items.reverse()
    return items


def generate_full_items(
    audio_name: str,
    audio_stem: str,
    conversation: Conversation,
    labels: Dict[int, str],
) -> List[dict]:
    utterances = conversation.utterances
    transcript = format_transcript_up_to(utterances, len(utterances), labels)
    return [{
        "input": {"transcript": transcript},
        "metadata": {
            "audio_stem": audio_stem,
            "audio_name": audio_name,
            "type": "full",
            "total_utterances": len(utterances),
            "conversation": audio_name,
            "offset_ms": utterances[0].start_ms if utterances else 0,
        },
        "expected_output": "",
    }]


def generate_stateful_incremental_items(
    audio_name: str,
    audio_stem: str,
    conversation: Conversation,
    labels: Dict[int, str],
    initial_state: Dict[str, Any],
    state_steps: List[Tuple[int, Dict[str, Any]]],
    customer_speaker_id: int,
) -> List[dict]:
    """Generate dataset items aligned with state step boundaries.

    For each step i in state_steps:
    - state = previous step's accumulated_state (or initial_state for step 0)
    - new_transcript = utterances from (prev_step_end + 1) to steps[i].utterance_index
    - old_transcript = up to CONTEXT_WINDOW utterances before new_transcript
    - expected_output = next seller utterance after the step boundary
    """
    utterances = conversation.utterances

    items: List[dict] = []
    for i, (utt_idx, acc_state) in enumerate(state_steps):
        # State is "outdated" — from the previous step (or initial_state for step 0)
        if i == 0:
            state = initial_state
            new_start = 0
        else:
            state = state_steps[i - 1][1]
            new_start = state_steps[i - 1][0] + 1
        new_end = utt_idx

        # Context window: up to CONTEXT_WINDOW utterances before new_start
        old_start = max(0, new_start - CONTEXT_WINDOW)
        old_end = new_start - 1

        old_transcript = ""
        if old_start <= old_end:
            old_transcript = format_utterance_range(utterances, old_start, old_end, labels)

        new_transcript = format_utterance_range(utterances, new_start, new_end, labels)
        if not new_transcript.strip():
            continue

        expected = _find_next_seller_text(utterances, new_end, customer_speaker_id)

        items.append({
            "input": {
                "state": state,
                "old_transcript": old_transcript,
                "new_transcript": new_transcript,
            },
            "metadata": {
                "audio_stem": audio_stem,
                "audio_name": audio_name,
                "type": "stateful_incremental",
                "utterance_index": utt_idx,
                "step": i,
                "conversation": audio_name,
            },
            "expected_output": expected,
        })

    return items


# ---------------------------------------------------------------------------
# Evaluation Sampler helpers
# ---------------------------------------------------------------------------

def format_conversation_for_selector(
    utterances: List[Utterance],
    labels: Dict[int, str],
    customer_speaker_id: int,
) -> str:
    """Build an indexed transcript with ``[CUSTOMER]`` markers.

    Each utterance is formatted as ``[idx] Speaker: text`` with customer
    utterances additionally marked ``[CUSTOMER]``.  No state or phase
    annotations — the LLM infers conversation dynamics from the dialogue.
    """
    lines: List[str] = []
    for idx, utt in enumerate(utterances):
        text = utt.cleaned_text
        if not text:
            continue
        label = labels.get(utt.speaker_id, f"Speaker {utt.speaker_id}")
        marker = " [CUSTOMER]" if utt.speaker_id == customer_speaker_id else ""
        lines.append(f"[{idx}]{marker} {label}: {text}")
    return "\n".join(lines)


def compile_selector_prompt(
    langfuse_client,
    prompt_name: str,
    transcript: str,
    target_count: int,
) -> Tuple[List[dict], dict]:
    """Fetch a prompt from Langfuse and build the message list.

    The Langfuse prompt (text or chat) provides the system instructions.
    The transcript and target count are always sent as a separate user
    message to guarantee the model receives the full conversation.

    Returns (messages, config).
    """
    prompt = langfuse_client.get_prompt(prompt_name)
    config = prompt.config or {}

    # Compile with variables (handles {{target_count}} etc. in the template)
    compiled = prompt.compile(
        transcript=transcript, target_count=str(target_count),
    )

    # Build the user message with the actual transcript
    user_content = (
        f"Here is the annotated sales call transcript:\n\n"
        f"{transcript}\n\n"
        f"Select {target_count} evaluation-worthy customer utterance indices.\n\n"
        f"CRITICAL REMINDER: You may ONLY select indices that have the "
        f"[CUSTOMER] marker. Double-check each index before including it."
    )

    if isinstance(compiled, str):
        # Text prompt — use compiled text as system, transcript as user
        messages = [
            {"role": "system", "content": compiled},
            {"role": "user", "content": user_content},
        ]
    else:
        # Chat prompt — use compiled messages, then append transcript
        messages = list(compiled)
        # Only append if the compiled messages don't already contain the
        # transcript (i.e. the prompt didn't have a {{transcript}} variable)
        has_transcript = any(transcript[:100] in m.get("content", "") for m in messages)
        if not has_transcript:
            messages.append({"role": "user", "content": user_content})

    return messages, config


def select_evaluation_indices(
    langfuse_client,
    prompt_name: str,
    formatted_transcript: str,
    target_count: int,
    audio_stem: str,
) -> List[dict]:
    """Use the LLM to select evaluation-worthy customer utterance indices.

    Returns a list of ``{"index": int, "reason": str, "phase": str}`` dicts.
    """
    messages, config = compile_selector_prompt(
        langfuse_client,
        prompt_name,
        formatted_transcript,
        target_count,
    )

    # Determine response_format from prompt config
    response_format = None
    rf = config.get("response_format")
    if rf == "json_object" or rf == {"type": "json_object"}:
        response_format = {"type": "json_object"}
    elif isinstance(rf, dict) and rf.get("type"):
        response_format = rf
    # Fall back: if config has a json_schema key, use json_object mode
    if response_format is None and config.get("json_schema"):
        response_format = {"type": "json_object"}

    llm = get_llm_client("fireworks", temperature=0.3, max_tokens=None)
    info(f"  Calling LLM for index selection on '{audio_stem}' ...")
    raw = llm.generate(messages=messages, response_format=response_format)

    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    data = json.loads(cleaned)
    selected = data.get("selected_indices", [])
    return selected


def lookup_state_before_index(
    index: int,
    initial_state: Dict[str, Any],
    state_steps: List[Tuple[int, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], int]:
    """Find the latest state step strictly before *index*.

    Returns ``(state, new_start)`` where *new_start* is the utterance after
    the matched step boundary.  Falls back to ``(initial_state, 0)`` when no
    prior step exists.
    """
    best_state = initial_state
    best_new_start = 0
    for step_idx, acc_state in state_steps:
        if step_idx < index:
            best_state = acc_state
            best_new_start = step_idx + 1
        else:
            break
    return best_state, best_new_start


def generate_evaluation_sample_items(
    audio_name: str,
    audio_stem: str,
    conversation: Conversation,
    labels: Dict[int, str],
    initial_state: Dict[str, Any],
    state_steps: List[Tuple[int, Dict[str, Any]]],
    customer_speaker_id: int,
    selected_indices: List[dict],
) -> List[dict]:
    """Generate dataset items for each LLM-selected evaluation index.

    Each item includes four input fields enabling side-by-side evaluation:
    - ``transcript`` — full transcript up to the customer utterance
    - ``state`` — latest state *before* the utterance
    - ``old_transcript`` — context window before the new segment
    - ``new_transcript`` — utterances from state boundary to the utterance
    """
    utterances = conversation.utterances
    items: List[dict] = []

    for sel in selected_indices:
        idx = sel["index"]
        reason = sel.get("reason", "")
        phase = sel.get("phase", "")

        if idx < 0 or idx >= len(utterances):
            continue

        # Full transcript up to and including this utterance
        transcript = format_transcript_up_to(utterances, idx + 1, labels)

        # State and segment boundary
        state, new_start = lookup_state_before_index(idx, initial_state, state_steps)

        # Context window before the new segment
        old_start = max(0, new_start - CONTEXT_WINDOW)
        old_end = new_start - 1
        old_transcript = ""
        if old_start <= old_end:
            old_transcript = format_utterance_range(utterances, old_start, old_end, labels)

        # New transcript segment
        new_transcript = format_utterance_range(utterances, new_start, idx, labels)

        # Expected output
        expected = _find_next_seller_text(utterances, idx, customer_speaker_id)

        items.append({
            "input": {
                "transcript": transcript,
                "state": state,
                "old_transcript": old_transcript,
                "new_transcript": new_transcript,
            },
            "metadata": {
                "audio_stem": audio_stem,
                "audio_name": audio_name,
                "type": "evaluation_sample",
                "utterance_index": idx,
                "phase": phase,
                "selection_reason": reason,
                "conversation": audio_name,
            },
            "expected_output": expected,
        })

    return items


# ---------------------------------------------------------------------------
# Langfuse upload
# ---------------------------------------------------------------------------

def create_langfuse_dataset(client, name: str, description: str, language: str = "Dansk") -> None:
    try:
        _with_rate_limit_backoff(
            lambda: client.create_dataset(
                name=name,
                description=description,
                metadata={"language": language},
            )
        )
        success(f"Created dataset: {name}")
    except Exception as exc:
        warn(f"create_dataset skipped (may already exist): {exc}")


def upload_items(client, dataset_name: str, items: List[dict], conversation_name: str) -> int:
    if not items:
        return 0

    created = 0
    total = len(items)
    for idx, item in enumerate(items):
        metadata = dict(item.get("metadata", {}))
        metadata["conversation"] = conversation_name
        time.sleep(0.5)
        try:
            _with_rate_limit_backoff(
                lambda item=item, metadata=metadata: client.create_dataset_item(
                    dataset_name=dataset_name,
                    input=item["input"],
                    metadata=metadata,
                    expected_output=item["expected_output"],
                )
            )
            created += 1
        except Exception as exc:
            error(f"Failed to push item {idx}: {exc}")
        progress(idx + 1, total, conversation_name)

    return created


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main() -> None:
    header("Callbuddy Dataset Generator")

    # 1. Select dataset type(s)
    type_choice = prompt_choice(
        "Which dataset type(s) do you want to generate?",
        ["1 - Incremental", "2 - Full", "3 - Stateful Incremental", "4 - Evaluation Sampler"],
    )
    selected_types: List[int] = []
    if type_choice.startswith("1"):
        selected_types = [1]
    elif type_choice.startswith("2"):
        selected_types = [2]
    elif type_choice.startswith("3"):
        selected_types = [3]
    elif type_choice.startswith("4"):
        selected_types = [4]

    info(f"Selected types: {selected_types}")

    # 2. Campaign path
    campaign_path = prompt_string(
        "Campaign path (relative to customer_data/)",
        prefix="customer_data/",
    )
    base_dir = (_PROJECT_ROOT / "customer_data" / campaign_path).resolve()
    if not base_dir.exists():
        error(f"Directory not found: {base_dir}")
        sys.exit(1)
    success(f"Using: {base_dir}")

    # 3. Load config + data
    config = load_config(base_dir)
    campaign_id = config.get("campaign", {}).get("id", base_dir.name)
    language = config.get("campaign", {}).get("language", "Dansk")
    speaker_map = load_speaker_map(base_dir)

    # 4. Dataset name
    dataset_name = ""
    if 1 in selected_types or 2 in selected_types or 4 in selected_types:
        dataset_name = prompt_string("Dataset name", default=campaign_id)

    # 5. Load conversations
    #    Each entry: (audio_name, audio_stem, conversation, file_stem)
    #    file_stem is the conversation filename without .json (e.g. "conv_1")
    conv_files = load_processed_conversations(base_dir)
    info(f"Found {len(conv_files)} processed conversation(s)")

    conversations: List[Tuple[str, str, Conversation, str]] = []
    for cf in conv_files:
        with cf.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        audio_name, audio_stem, conv = parse_conversation(raw)
        conversations.append((audio_name, audio_stem, conv, cf.stem))

    # 6. Type-specific parameters
    min_time_ms = 0

    if 1 in selected_types:
        while True:
            min_time_ms = prompt_int("Min time between customer utterances (ms)", default=0)
            # Show count
            total_count = 0
            for audio_name, audio_stem, conv, _ in conversations:
                cid = get_customer_speaker_id(audio_stem, speaker_map)
                items = generate_incremental_items(audio_name, audio_stem, conv, speaker_label_map(audio_stem, speaker_map), min_time_ms, cid)
                total_count += len(items)
            info(f"With min_time_ms={min_time_ms}: {C.BOLD}{total_count}{C.RESET} incremental items across {len(conversations)} conversation(s)")
            if prompt_confirm("Use this setting?"):
                break

    # Type 4 parameters
    selector_prompt_name = "case_identifier"
    eval_target_count = 10

    if 4 in selected_types:
        selector_prompt_name = prompt_string(
            "Langfuse selector prompt name",
            default="case_identifier",
        )
        eval_target_count = prompt_int("Target indices per conversation", default=10)

    # stateful_datasets: list of (ds_name, audio_name, audio_stem, conv, initial_state, steps, item_count) for type 3
    stateful_datasets: List[Tuple[str, str, str, Conversation, Dict[str, Any], List[Tuple[int, Dict[str, Any]]], int]] = []

    if 3 in selected_types:
        # Auto-detect state_experiments/ from campaign directory
        state_exp_dir = base_dir / "state_experiments"
        if not state_exp_dir.exists():
            error(f"No state_experiments/ directory found in {base_dir}")
            sys.exit(1)

        try:
            experiment_entries = load_experiment_state_dir(str(state_exp_dir))
            if not experiment_entries:
                error(f"No .state.json files found in {state_exp_dir}")
                sys.exit(1)
            success(f"Loaded {len(experiment_entries)} .state.json file(s) from {state_exp_dir.name}/")
        except Exception as exc:
            error(f"Failed to load state experiments: {exc}")
            sys.exit(1)

        # Match experiments to conversations, build dataset preview
        for exp_audio_stem, exp_initial_state, exp_steps in experiment_entries:
            match = None
            for audio_name, audio_stem, conv, file_stem in conversations:
                if audio_stem == exp_audio_stem:
                    match = (audio_name, audio_stem, conv, file_stem)
                    break
            if not match:
                warn(f"No conversation matches audio_stem='{exp_audio_stem}', skipping")
                continue
            m_audio_name, m_audio_stem, m_conv, _ = match
            labels = speaker_label_map(m_audio_stem, speaker_map)
            cid = get_customer_speaker_id(m_audio_stem, speaker_map)
            preview_items = generate_stateful_incremental_items(
                m_audio_name, m_audio_stem, m_conv, labels, exp_initial_state, exp_steps, cid,
            )
            ds_name = f"{m_audio_stem}_stateful_incremental"
            stateful_datasets.append((ds_name, m_audio_name, m_audio_stem, m_conv, exp_initial_state, exp_steps, len(preview_items)))

    # evaluation_entries: list of (audio_name, audio_stem, conv, initial_state, steps, selected_indices) for type 4
    evaluation_entries: List[Tuple[str, str, Conversation, Dict[str, Any], List[Tuple[int, Dict[str, Any]]], List[dict]]] = []

    if 4 in selected_types:
        # Load state experiments
        state_exp_dir = base_dir / "state_experiments"
        if not state_exp_dir.exists():
            error(f"No state_experiments/ directory found in {base_dir}")
            sys.exit(1)

        try:
            experiment_entries = load_experiment_state_dir(str(state_exp_dir))
            if not experiment_entries:
                error(f"No .state.json files found in {state_exp_dir}")
                sys.exit(1)
            success(f"Loaded {len(experiment_entries)} .state.json file(s) from {state_exp_dir.name}/")
        except Exception as exc:
            error(f"Failed to load state experiments: {exc}")
            sys.exit(1)

        # Init Langfuse early — needed for prompt fetch before the confirm step
        info("Initializing Langfuse (needed for selector prompt) ...")
        langfuse_client_early = init_langfuse(push_to_langfuse=True)
        if langfuse_client_early is None:
            error("Could not initialize Langfuse. Check your .env credentials.")
            sys.exit(1)

        # Match experiments to conversations, run LLM selection
        for exp_audio_stem, exp_initial_state, exp_steps in experiment_entries:
            match = None
            for audio_name, audio_stem, conv, file_stem in conversations:
                if audio_stem == exp_audio_stem:
                    match = (audio_name, audio_stem, conv, file_stem)
                    break
            if not match:
                warn(f"No conversation matches audio_stem='{exp_audio_stem}', skipping")
                continue
            m_audio_name, m_audio_stem, m_conv, _ = match
            labels = speaker_label_map(m_audio_stem, speaker_map)
            cid = get_customer_speaker_id(m_audio_stem, speaker_map)

            formatted = format_conversation_for_selector(
                m_conv.utterances, labels, cid,
            )
            try:
                selected = select_evaluation_indices(
                    langfuse_client_early, selector_prompt_name,
                    formatted, eval_target_count, m_audio_stem,
                )
            except Exception as exc:
                error(f"LLM selection failed for '{m_audio_stem}': {exc}")
                continue

            # Validate: keep only indices that are actual customer utterances within bounds
            valid = []
            for sel in selected:
                idx = sel.get("index")
                if idx is None:
                    continue
                idx = int(idx)
                if 0 <= idx < len(m_conv.utterances) and m_conv.utterances[idx].speaker_id == cid:
                    sel["index"] = idx
                    valid.append(sel)
                else:
                    warn(f"  Skipping index {idx} (not a valid customer utterance)")

            info(f"  '{m_audio_stem}': {len(valid)} valid indices selected:")
            for sel in valid:
                print(f"    [{sel['index']}] phase={sel.get('phase', '?')} — {sel.get('reason', '')}")

            evaluation_entries.append((m_audio_name, m_audio_stem, m_conv, exp_initial_state, exp_steps, valid))

    # 7. Summary + confirm
    header("Generation Summary")
    print(f"  Campaign:      {C.CYAN}{campaign_id}{C.RESET}")
    print(f"  Base dir:      {base_dir}")
    print(f"  Types:         {selected_types}")
    print(f"  Conversations: {len(conversations)}")
    if 1 in selected_types:
        print(f"  Dataset name:  {C.CYAN}{dataset_name}{C.RESET}")
        print(f"  min_time_ms:   {min_time_ms}")
    if 3 in selected_types:
        print(f"\n  Datasets to create ({len(stateful_datasets)}):")
        total_items = 0
        for ds_name, _, audio_stem, _, _, _, item_count in stateful_datasets:
            print(f"    {C.CYAN}{ds_name}{C.RESET}  ({item_count} items)")
            total_items += item_count
        print(f"  Total items:   {C.BOLD}{total_items}{C.RESET}")
    if 4 in selected_types:
        eval_total = sum(len(e[5]) for e in evaluation_entries)
        print(f"\n  Evaluation Sampler:")
        print(f"    Dataset:     {C.CYAN}{dataset_name}_evaluation_sample{C.RESET}")
        print(f"    Conversations: {len(evaluation_entries)}")
        for m_audio_name, m_audio_stem, _, _, _, sel_indices in evaluation_entries:
            print(f"      {m_audio_stem}: {len(sel_indices)} indices")
        print(f"    Total items: {C.BOLD}{eval_total}{C.RESET}")
    print()

    if not prompt_confirm("Generate and upload to Langfuse?"):
        info("Aborted.")
        sys.exit(0)

    # 8. Init Langfuse (reuse early client if type 4 already created one)
    if 4 in selected_types:
        langfuse_client = langfuse_client_early
    else:
        langfuse_client = init_langfuse(push_to_langfuse=True)
        if langfuse_client is None:
            error("Could not initialize Langfuse. Check your .env credentials.")
            sys.exit(1)
    success("Langfuse initialized")

    # 9. Generate + upload per type
    #    Type 1: one dataset per conversation
    #    Type 2: one shared dataset for all conversations
    #    Type 3: one dataset per conversation (from state files)
    for dtype in selected_types:
        suffix = {1: "incremental", 2: "full", 3: "stateful_incremental", 4: "evaluation_sample"}[dtype]

        header(f"Type {dtype}: {suffix}")

        if dtype == 2:
            # Full: single shared dataset
            ds_name = f"{dataset_name}_{suffix}"
            create_langfuse_dataset(
                langfuse_client,
                ds_name,
                f"{suffix.capitalize()} dataset for {campaign_id}",
                language,
            )

            total_uploaded = 0
            for audio_name, audio_stem, conv, file_stem in conversations:
                labels = speaker_label_map(audio_stem, speaker_map)
                items = generate_full_items(audio_name, audio_stem, conv, labels)

                if not items:
                    warn(f"No items for {file_stem}")
                    continue

                info(f"{file_stem}: {len(items)} items")
                created = upload_items(langfuse_client, ds_name, items, audio_name)
                total_uploaded += created

            success(f"Uploaded {total_uploaded} items to '{ds_name}'")

        elif dtype == 1:
            # Incremental: one dataset per conversation
            total_uploaded = 0
            for audio_name, audio_stem, conv, file_stem in conversations:
                ds_name = f"{file_stem}_{suffix}"
                labels = speaker_label_map(audio_stem, speaker_map)
                cid = get_customer_speaker_id(audio_stem, speaker_map)
                items = generate_incremental_items(audio_name, audio_stem, conv, labels, min_time_ms, cid)

                if not items:
                    warn(f"No items for {file_stem}")
                    continue

                create_langfuse_dataset(
                    langfuse_client,
                    ds_name,
                    f"{suffix.capitalize()} dataset for {file_stem}",
                    language,
                )

                info(f"{file_stem}: {len(items)} items -> '{ds_name}'")
                created = upload_items(langfuse_client, ds_name, items, audio_name)
                total_uploaded += created

            success(f"Uploaded {total_uploaded} total {suffix} items across {len(conversations)} dataset(s)")

        elif dtype == 3:
            # Stateful Incremental: one dataset per conversation
            total_uploaded = 0
            for ds_name, m_audio_name, m_audio_stem, m_conv, exp_initial_state, exp_steps, _ in stateful_datasets:
                labels = speaker_label_map(m_audio_stem, speaker_map)
                cid = get_customer_speaker_id(m_audio_stem, speaker_map)
                items = generate_stateful_incremental_items(
                    m_audio_name, m_audio_stem, m_conv, labels, exp_initial_state, exp_steps, cid,
                )

                if not items:
                    warn(f"No items for {m_audio_stem}")
                    continue

                create_langfuse_dataset(
                    langfuse_client,
                    ds_name,
                    f"Stateful incremental dataset for {m_audio_stem}",
                    language,
                )

                info(f"{m_audio_stem}: {len(items)} items -> '{ds_name}'")
                created = upload_items(langfuse_client, ds_name, items, m_audio_name)
                total_uploaded += created

            success(f"Uploaded {total_uploaded} total {suffix} items across {len(stateful_datasets)} dataset(s)")

        elif dtype == 4:
            # Evaluation Sampler: one shared dataset
            ds_name = f"{dataset_name}_{suffix}"
            create_langfuse_dataset(
                langfuse_client,
                ds_name,
                f"Evaluation sample dataset for {campaign_id}",
                language,
            )

            total_uploaded = 0
            for m_audio_name, m_audio_stem, m_conv, exp_initial_state, exp_steps, sel_indices in evaluation_entries:
                labels = speaker_label_map(m_audio_stem, speaker_map)
                cid = get_customer_speaker_id(m_audio_stem, speaker_map)
                items = generate_evaluation_sample_items(
                    m_audio_name, m_audio_stem, m_conv, labels,
                    exp_initial_state, exp_steps, cid, sel_indices,
                )

                if not items:
                    warn(f"No items for {m_audio_stem}")
                    continue

                info(f"{m_audio_stem}: {len(items)} items -> '{ds_name}'")
                created = upload_items(langfuse_client, ds_name, items, m_audio_name)
                total_uploaded += created

            success(f"Uploaded {total_uploaded} total {suffix} items to '{ds_name}'")

    header("Done!")
    success("All datasets generated and uploaded.")


if __name__ == "__main__":
    main()
