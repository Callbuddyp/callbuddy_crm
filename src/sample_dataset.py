#!/usr/bin/env python3
"""Sample a subset of dataset items by action type.

Usage:
    python sample_dataset.py \
        --customer-data-dir /path/to/customer \
        --type objection \
        --amount 10 \
        --name "objection_test_v1" \
        --push-to-langfuse true
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

from models.conversation import Conversation
from models.scenario_test_case import ScenarioTestCase
from services.langfuse import ensure_dataset, init_langfuse, push_dataset_items
from services.prompts import ACTION_TYPES

TESTCASE_DIRNAME = "dataset"
TRANSCRIPT_DIRNAME = "processed_conversations"
DEFAULT_LANGUAGE = "Dansk"


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a subset of dataset items by action type."
    )
    parser.add_argument(
        "--customer-data-dir",
        required=True,
        help="Path to customer data directory containing dataset/ folder.",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=ACTION_TYPES,
        help=f"Action type to sample. Options: {ACTION_TYPES}",
    )
    parser.add_argument(
        "--amount",
        type=int,
        required=True,
        help="Number of cases to sample.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Name for the sampled dataset (used in Langfuse and output file).",
    )
    parser.add_argument(
        "--push-to-langfuse",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Upload sampled cases to Langfuse.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save sampled cases. Defaults to dataset/sampled/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_speaker_map(base_dir: Path) -> dict:
    path = base_dir / "speaker_map.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"Warning: could not load speaker map {path}: {exc}")
        return {}


def speaker_label_map(audio_stem: str, speaker_map: dict) -> dict[int, str]:
    speaker1_is_seller = speaker_map.get(audio_stem)
    if not isinstance(speaker1_is_seller, bool):
        speaker1_is_seller = True
    return {
        1: "Sælger" if speaker1_is_seller else "Kunden",
        2: "Kunden" if speaker1_is_seller else "Sælger",
    }


def load_cases_file(path: Path) -> tuple[str, str, List[ScenarioTestCase]]:
    """Load cases from a dataset file, returning (audio_name, audio_stem, cases)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    audio_name = data.get("audio_name", path.stem)
    audio_stem = data.get("audio_stem", path.stem.replace(".mp3", "").replace(".wav", ""))
    raw_cases = data.get("cases") or []
    
    if not isinstance(raw_cases, list):
        return audio_name, audio_stem, []

    cases: List[ScenarioTestCase] = []
    for case_data in raw_cases:
        try:
            cases.append(ScenarioTestCase(**case_data))
        except Exception:
            pass
    return audio_name, audio_stem, cases


def load_transcription(path: Path) -> Conversation:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    conv_dict = data.get("conversation") or data
    return Conversation.from_dict(conv_dict)


def format_transcript_snippet(conversation: Conversation, labels: dict[int, str], anchor_utterance_index: int) -> str:
    def label_for(speaker_id: int) -> str:
        return labels.get(speaker_id, f"Speaker {speaker_id}")

    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal parts, current_speaker, buffer
        if current_speaker is None or not buffer:
            return
        label = label_for(current_speaker)
        parts.append(f"{label}:\n\n{' '.join(buffer)}")

    for index, utt in enumerate(conversation.utterances):
        if index > anchor_utterance_index:
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
    return "\n\n".join(parts)


def build_dataset_item(
    audio_name: str,
    audio_stem: str,
    conversation: Conversation,
    case: ScenarioTestCase,
    label_map: dict[int, str],
) -> dict:
    transcript_snippet = format_transcript_snippet(conversation, label_map, case.anchor_utterance_index)
    try:
        expected_utterance = conversation.get_utterance(case.anchor_utterance_index + 1)
        expected_output = expected_utterance.cleaned_text
    except Exception:
        expected_output = ""

    return {
        "input": {
            "transcript": transcript_snippet,
            "suggested_tests": case.suggested_tests,
        },
        "metadata": {
            "audio_stem": audio_stem,
            "audio_name": audio_name,
            "reason": case.reason,
            "customer_text": case.customer_text,
            "seller_action": case.seller_action,
        },
        "expected_output": expected_output,
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.customer_data_dir).expanduser().resolve()
    case_dir = base_dir / TESTCASE_DIRNAME
    transcript_dir = base_dir / TRANSCRIPT_DIRNAME
    
    if not case_dir.exists():
        raise ValueError(f"Dataset directory not found: {case_dir}")

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Load speaker map
    speaker_map = load_speaker_map(base_dir)

    # Collect all cases matching the requested type
    print(f"Scanning for '{args.type}' cases in {case_dir}...")
    matching_cases: List[tuple[str, str, ScenarioTestCase, Path]] = []
    
    for case_file in sorted(case_dir.glob("*.json")):
        audio_name, audio_stem, cases = load_cases_file(case_file)
        for case in cases:
            # Check if this case matches the requested type
            if args.type in case.suggested_tests:
                matching_cases.append((audio_name, audio_stem, case, case_file))

    print(f"Found {len(matching_cases)} total '{args.type}' cases.")

    if not matching_cases:
        print("No matching cases found. Exiting.")
        return

    # Sample the requested amount
    sample_size = min(args.amount, len(matching_cases))
    if sample_size < args.amount:
        print(f"Warning: Requested {args.amount} but only {len(matching_cases)} available.")
    
    sampled = random.sample(matching_cases, sample_size)
    print(f"Sampled {len(sampled)} cases.")

    # Build dataset items
    items: List[dict] = []
    for audio_name, audio_stem, case, case_file in sampled:
        # Find corresponding transcript
        transcript_path = transcript_dir / f"{audio_name}.json"
        if not transcript_path.exists():
            # Try without extension duplication
            for possible_name in [audio_name, f"{audio_stem}.mp3", f"{audio_stem}.wav"]:
                alt_path = transcript_dir / f"{possible_name}.json"
                if alt_path.exists():
                    transcript_path = alt_path
                    break
        
        if not transcript_path.exists():
            print(f"  Warning: Could not find transcript for {audio_name}, skipping.")
            continue
        
        conversation = load_transcription(transcript_path)
        label_map = speaker_label_map(audio_stem, speaker_map)
        item = build_dataset_item(audio_name, audio_stem, conversation, case, label_map)
        items.append(item)

    print(f"Built {len(items)} dataset items.")

    # Save locally
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "sampled"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.name}.json"
    
    output_data = {
        "name": args.name,
        "type": args.type,
        "amount_requested": args.amount,
        "amount_sampled": len(items),
        "seed": args.seed,
        "items": items,
    }
    
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Saved sampled dataset to: {output_file}")

    # Push to Langfuse if requested
    if args.push_to_langfuse:
        langfuse_client = init_langfuse(push_to_langfuse=True)
        if langfuse_client:
            ensure_dataset([args.type], langfuse_client, args.name, DEFAULT_LANGUAGE)
            created = push_dataset_items(
                langfuse_client=langfuse_client,
                dataset_name=args.name,
                items=items,
                conversation_name=args.name,
            )
            print(f"Pushed {created} items to Langfuse dataset '{args.name}'.")
        else:
            print("Warning: Could not initialize Langfuse client.")

    print("Done!")


if __name__ == "__main__":
    main()
