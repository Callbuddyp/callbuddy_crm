#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from models.conversation import Conversation
from models.scenario_test_case import ScenarioTestCase
from services.langfuse import ensure_dataset, init_langfuse, push_dataset_items
from services.prompts import ACTION_TYPES
from services.soniox import build_soniox_session, transcribe_audio
from services.vad import VADConfig, detect_ai_suggestion_utterances
from utils import load_env_value

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
TRANSCRIPT_DIRNAME = "processed_conversations"
TESTCASE_DIRNAME = "dataset"
DEFAULT_LANGUAGE = "Dansk"
DEFAULT_LANGUAGE_HINTS: Optional[List[str]] = None
DEFAULT_KEEP_SONIOX = False
DEFAULT_MAX_FILES: Optional[int] = None

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


def load_customer_config(base_dir: Path) -> dict:
    """Load customer configuration from config.json in the customer data directory.
    
    Supports both old flat format and new nested format:
    - Old: {"name": "x", "action_types": [...]}
    - New: {"campaign": {"id": "x", "name": "y"}, "dataset": {"action_types": [...]}}
    
    Returns a dict with at least 'action_types' and 'name' keys.
    Falls back to default ACTION_TYPES if config is missing or invalid.
    """
    path = base_dir / "config.json"
    if not path.exists():
        print(f"No config.json found in {base_dir}, using default action types.")
        return {"action_types": ACTION_TYPES, "name": base_dir.name}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"Warning: config.json is not a valid object, using defaults.")
            return {"action_types": ACTION_TYPES, "name": base_dir.name}
        
        # Handle new nested format
        if "campaign" in data or "dataset" in data:
            campaign = data.get("campaign", {})
            dataset = data.get("dataset", {})
            vad = data.get("vad", {})
            
            # Extract values from nested structure
            name = campaign.get("id") or campaign.get("name") or base_dir.name
            action_types = dataset.get("action_types", ACTION_TYPES)
            
            # Build result in old format for compatibility
            result = {
                "name": name,
                "action_types": action_types,
                "vad_config": {
                    "speech_minimum_duration_ms": vad.get("speech_minimum_duration_ms", 2000),
                    "silence_before_ai_ms": vad.get("silence_before_ai_ms", 500),
                },
            }
        else:
            # Handle old flat format
            result = dict(data)
            if "name" not in result:
                result["name"] = base_dir.name
            action_types = result.get("action_types", ACTION_TYPES)
        
        # Validate action_types
        if not isinstance(action_types, list):
            print(f"Warning: action_types is not a list, using defaults.")
            action_types = ACTION_TYPES
        
        # Filter to only valid action types
        valid_types = [t for t in action_types if t in ACTION_TYPES]
        if len(valid_types) != len(action_types):
            invalid = set(action_types) - set(valid_types)
            print(f"Warning: ignoring invalid action types: {invalid}")
        
        if not valid_types:
            print(f"Warning: no valid action types found, using defaults.")
            valid_types = ACTION_TYPES
        
        result["action_types"] = valid_types
        return result
    except Exception as exc:
        print(f"Warning: could not load config {path}: {exc}")
        return {"action_types": ACTION_TYPES, "name": base_dir.name}


def speaker_label_map(audio_stem: str, speaker_map: dict) -> dict[int, str]:
    speaker1_is_seller = speaker_map.get(audio_stem)
    if not isinstance(speaker1_is_seller, bool):
        speaker1_is_seller = True
    return {
        1: "Sælger" if speaker1_is_seller else "Kunden",
        2: "Kunden" if speaker1_is_seller else "Sælger",
    }


def format_transcript_snippet(conversation: Conversation, labels: dict[int, str], ancher_utterance_index: int) -> str:
    def label_for(speaker_id: int) -> str:
        return labels.get(speaker_id, f"Speaker {speaker_id}")

    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        if current_speaker is None or not buffer:
            return
        label = label_for(current_speaker)
        parts.append(f"{label}:\n\n{' '.join(buffer)}")

    for index, utt in enumerate(conversation.utterances):
        # Only include transcription up until the current index
        if index > ancher_utterance_index:
            break;

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
        description="Generate scenario test cases for a folder of conversations."
    )
    parser.add_argument(
        "--customer-data-dir",
        required=True,
        help="Path containing conversations/, processed_conversations/, dataset/ folders.",
    )
    parser.add_argument(
        "--push-to-langfuse",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Upload generated cases to Langfuse.",
    )
    parser.add_argument(
        "--review-scenarios",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Interactively review generated cases before saving.",
    )
    parser.add_argument(
        "--reupload",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="If cases already exist, reload them and push them to Langfuse again.",
    )
    return parser.parse_args()


def audio_files(conv_dir: Path, max_files: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for path in sorted(conv_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(path)
            if max_files is not None and len(files) >= max_files:
                break
    return files


def load_transcription(path: Path) -> Tuple[Conversation, Optional[dict]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    conv_dict = data.get("conversation") or data
    conversation = Conversation.from_dict(conv_dict)
    return conversation, data.get("soniox")


def load_cases_file(path: Path) -> List[ScenarioTestCase]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw_cases = data.get("cases") or []
    if not isinstance(raw_cases, list):
        print(f"Warning: invalid cases format in {path.name}, expected a list.")
        return []

    cases: List[ScenarioTestCase] = []
    for idx, case_data in enumerate(raw_cases):
        try:
            cases.append(ScenarioTestCase(**case_data))
        except Exception as exc:
            print(f"Warning: could not parse case #{idx} in {path.name}: {exc}")
    return cases


def save_transcription(
    path: Path,
    audio_path: Path,
    conversation: Conversation,
    soniox_metadata: Optional[dict],
) -> None:
    entry = {
        "audio_name": audio_path.name,
        "audio_stem": audio_path.stem,
        "conversation": conversation.to_dict(),
    }
    if soniox_metadata:
        entry["soniox"] = soniox_metadata
    with path.open("w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)


def save_cases(path: Path, audio_path: Path, cases: List[ScenarioTestCase]) -> None:
    payload = {
        "audio_name": audio_path.name,
        "audio_stem": audio_path.stem,
        "cases": [case.dict() for case in cases],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def review_cases(audio_path: Path, conversation: Conversation, cases: List[ScenarioTestCase]) -> str:
    print(f"\nTest cases for {audio_path.name}:")
    if not cases:
        print("  -> Model returned no cases.")
        return "skip"

    for idx, case in enumerate(cases, start=1):
        try:
            utterance = conversation.get_utterance(case.anchor_utterance_index)
            utt_text = utterance.cleaned_text
            speaker = utterance.speaker_id
        except Exception:
            utt_text = "<invalid anchor index>"
            speaker = "?"
        tests = ", ".join(case.suggested_tests)
        print(f"- [{idx}] utterance #{case.anchor_utterance_index} (speaker {speaker}): {utt_text}")
        print(f"    suggested_tests: {tests}")

    while True:
        choice = input("Save these cases? [y]es/[s]kip/[a]bort: ").strip().lower()
        if choice in ("", "y", "yes"):
            return "save"
        if choice in ("s", "skip", "n", "no"):
            return "skip"
        if choice in ("a", "abort", "q", "quit", "exit"):
            return "abort"
        print("Please enter y, s, or a.")


def build_dataset_items(
    audio_path: Path,
    conversation: Conversation,
    cases: List[ScenarioTestCase],
    label_map: dict[int, str],
) -> List[dict]:
    items = []
    for case in cases:
        transcript_snippet = format_transcript_snippet(conversation, label_map, case.anchor_utterance_index)
        try:
            expected_utterance = conversation.get_utterance(case.anchor_utterance_index + 1)
            expected_output = expected_utterance.cleaned_text
        except Exception:
            expected_output = ""

        items.append(
            {
                "input": {
                    "transcript": transcript_snippet,
                    "suggested_tests": case.suggested_tests,
                },
                "metadata": {
                    "audio_stem": audio_path.stem,
                    "audio_name": audio_path.name,
                    "reason": case.reason,
                    "customer_text": case.customer_text,
                    "seller_action": case.seller_action,
                },
                "expected_output": expected_output,
            }
        )
    return items


def main() -> None:
    args = parse_args()
    base_dir = Path(args.customer_data_dir).expanduser().resolve()
    audio_dir = base_dir / "conversations"
    transcript_dir = base_dir / TRANSCRIPT_DIRNAME
    case_dir = base_dir / TESTCASE_DIRNAME
    langfuse_client = init_langfuse(args.push_to_langfuse)

    # Load customer config for action types
    customer_config = load_customer_config(base_dir)
    types = customer_config["action_types"]
    customer_name = customer_config.get("name", base_dir.name)
    
    # Dataset name is derived from the config name (or directory name as fallback)
    dataset_name = customer_name
    
    print(f"Customer: {customer_name}")
    print(f"Action types: {types}")
    print(f"Dataset name: {dataset_name}")

    if not audio_dir.exists():
        raise ValueError(f"Conversation directory not found: {audio_dir}")

    transcript_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    baseten_api_key = os.getenv("BASETEN_API_KEY") or load_env_value("BASETEN_API_KEY")
    if not baseten_api_key:
        raise ValueError("Baseten API key is required (set BASETEN_API_KEY).")

    if args.push_to_langfuse and langfuse_client:
        ensure_dataset(types, langfuse_client, dataset_name, DEFAULT_LANGUAGE)

    speaker_map = load_speaker_map(base_dir)
    files = audio_files(audio_dir, DEFAULT_MAX_FILES)
    if not files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(files)} audio files.")
    soniox_session = None

    for audio_path in files:
        print(f"\nProcessing {audio_path.name}...")
        transcript_path = transcript_dir / f"{audio_path.name}.json"
        cases_path = case_dir / f"{audio_path.name}.json"

        if cases_path.exists():
            print("  -> Test cases already exist, skipping generation.")
            if args.reupload:
                if not args.push_to_langfuse or not langfuse_client:
                    print("  -> Reupload requested, but Langfuse upload is disabled.")
                elif not transcript_path.exists():
                    print("  -> Cannot reupload: cached transcription is missing.")
                else:
                    try:
                        conversation, _ = load_transcription(transcript_path)
                    except Exception as exc:
                        print(f"  -> Failed to load transcription for reupload: {exc}")
                    else:
                        existing_cases = load_cases_file(cases_path)
                        if not existing_cases:
                            print("  -> No valid cases found to reupload.")
                        else:
                            label_map = speaker_label_map(audio_path.stem, speaker_map)
                            items = build_dataset_items(audio_path, conversation, existing_cases, label_map)
                            created = push_dataset_items(
                                langfuse_client=langfuse_client,
                                dataset_name=dataset_name,
                                items=items,
                                conversation_name=audio_path.name,
                            )
                            print(f"  -> Reuploaded {created} dataset items to Langfuse.")
            continue

        if transcript_path.exists():
            conversation, soniox_metadata = load_transcription(transcript_path)
            print("  -> Loaded cached transcription.")
        else:
            soniox_session = soniox_session or build_soniox_session()
            transcription_result = transcribe_audio(
                session=soniox_session,
                audio_path=audio_path,
                language_hints=DEFAULT_LANGUAGE_HINTS,
                keep_soniox=DEFAULT_KEEP_SONIOX,
            )
            conversation = Conversation.from_soniox_transcription(
                transcription_result.transcription, language=DEFAULT_LANGUAGE
            )
            soniox_metadata = {
                "file_id": transcription_result.file_id,
                "transcription_id": transcription_result.transcription_id,
                "kept_on_soniox": transcription_result.kept_on_soniox,
                "language_hints": transcription_result.language_hints,
                "translation": transcription_result.translation,
            }
            save_transcription(transcript_path, audio_path, conversation, soniox_metadata)
            print("  -> Saved new transcription.")

        # Run VAD for ai_suggestion detection if enabled
        vad_suggestion_indices = None
        if "ai_suggestion" in types:
            vad_config = VADConfig.from_dict(customer_config.get("vad_config"))
            vad_suggestion_indices = detect_ai_suggestion_utterances(
                audio_path=audio_path,
                utterances=conversation.to_dict()["utterances"],
                config=vad_config,
            )

        cases = conversation.generate_test_cases(
            is_speaker_one_seller=speaker_map[audio_path.stem],
            audio_name=audio_path.name,
            audio_stem=audio_path.stem,
            generate_ai_cases=False,
            soniox_metadata=soniox_metadata,
            action_types=types,
            vad_suggestion_indices=vad_suggestion_indices,
        )

        

        decision = review_cases(audio_path, conversation, cases) if args.review_scenarios else "save"
        if decision == "abort":
            print("Aborted by user.")
            break
        if decision == "skip":
            print("  -> Skipped saving cases.")
            continue

        save_cases(cases_path, audio_path, cases)
        print(f"  -> Saved {len(cases)} test cases.")
        if args.push_to_langfuse and langfuse_client:
            label_map = speaker_label_map(audio_path.stem, speaker_map)
            items = build_dataset_items(audio_path, conversation, cases, label_map)
            created = push_dataset_items(
                langfuse_client=langfuse_client,
                dataset_name=dataset_name,
                items=items,
                conversation_name=audio_path.name,
            )
            print(f"  -> Pushed {created} dataset items to Langfuse.")


if __name__ == "__main__":
    main()
