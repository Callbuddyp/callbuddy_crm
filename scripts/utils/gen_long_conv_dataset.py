#!/usr/bin/env python3
"""
Generate a dataset from conversations for testing seller responses.

This script creates dataset items where each item contains all transcription
up until the point where the seller is supposed to speak. This allows testing
the model's ability to generate appropriate seller responses at each turn.

Usage:
    python scripts/gen_long_conv_dataset.py --conversation-path <path_to_conversation.json>
    python scripts/gen_long_conv_dataset.py --conversation-path <path_to_conversation.json> --output <output.json>
    python scripts/gen_long_conv_dataset.py --conversation-path <file.json> --push-to-langfuse --dataset-name my_dataset
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from services.langfuse import init_langfuse, push_dataset_items


def load_conversation(path: Path) -> dict:
    """Load a conversation JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def determine_seller_speaker_id(
    utterances: List[dict],
    speaker_map: Optional[dict] = None,
    audio_stem: Optional[str] = None,
) -> int:
    """
    Determine which speaker_id corresponds to the seller.
    
    If speaker_map is provided and contains mapping for the audio_stem,
    use that. Otherwise, default to speaker_id 2 being seller (common pattern).
    """
    if speaker_map and audio_stem:
        speaker1_is_seller = speaker_map.get(audio_stem)
        if isinstance(speaker1_is_seller, bool):
            return 1 if speaker1_is_seller else 2
    
    # Default: speaker 2 is typically the seller (caller)
    return 2


def get_speaker_label(speaker_id: int, seller_speaker_id: int) -> str:
    """Get the label for a speaker based on whether they are the seller or buyer."""
    if speaker_id == seller_speaker_id:
        return "Sælger"
    return "Kunden"


def format_transcript_up_to(
    utterances: List[dict],
    end_index: int,
    seller_speaker_id: int,
) -> str:
    """
    Format transcript from utterances up to (but not including) end_index.
    
    Groups consecutive utterances by the same speaker and formats them
    with proper labels (Sælger/Kunden).
    """
    parts: List[str] = []
    current_speaker: Optional[int] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal parts, current_speaker, buffer
        if current_speaker is None or not buffer:
            return
        label = get_speaker_label(current_speaker, seller_speaker_id)
        parts.append(f"{label}:\n\n{' '.join(buffer)}")
        buffer = []

    for index, utt in enumerate(utterances):
        if index >= end_index:
            break

        text = utt.get("text", "").strip()
        if not text:
            continue

        speaker_id = utt.get("speaker_id")
        if speaker_id == current_speaker:
            buffer.append(text)
            continue

        flush()
        current_speaker = speaker_id
        buffer = [text]

    flush()
    return "\n\n".join(parts)


def generate_long_conversation_dataset(
    conversation_data: dict,
    speaker_map: Optional[dict] = None,
) -> dict:
    """
    Generate a dataset with items for each seller turn.
    
    Each item contains:
    - input: transcript up to the point where seller should speak
    - expected_output: what the seller actually said
    - metadata: information about the conversation and position
    """
    audio_name = conversation_data.get("audio_name", "unknown")
    audio_stem = conversation_data.get("audio_stem", audio_name.rsplit(".", 1)[0] if "." in audio_name else audio_name)
    
    conv = conversation_data.get("conversation", {})
    utterances = conv.get("utterances", [])
    language = conv.get("language", "Dansk")
    
    if not utterances:
        return {
            "name": f"long_conversation_{audio_stem}",
            "type": "long_conversation",
            "source_audio": audio_name,
            "language": language,
            "items": [],
        }
    
    # Determine seller speaker ID
    seller_speaker_id = determine_seller_speaker_id(utterances, speaker_map, audio_stem)
    
    items = []
    
    for index, utt in enumerate(utterances):
        speaker_id = utt.get("speaker_id")
        
        # We only want to create dataset items where the seller is about to speak
        if speaker_id != seller_speaker_id:
            continue
        
        # Skip the first utterance if seller starts (need some context)
        if index == 0:
            continue
        
        # Get all transcript up to this point (not including current utterance)
        transcript = format_transcript_up_to(utterances, index, seller_speaker_id)
        
        # The expected output is what the seller actually said
        expected_output = utt.get("text", "").strip()
        
        # Get the previous utterance (what prompted the seller to respond)
        prev_utt = utterances[index - 1] if index > 0 else None
        customer_text = prev_utt.get("text", "").strip() if prev_utt else ""
        
        item = {
            "input": {
                "transcript": transcript,
                "suggested_tests": ["seller_response"],
            },
            "metadata": {
                "audio_stem": audio_stem,
                "audio_name": audio_name,
                "utterance_index": index,
                "customer_text": customer_text,
                "start_ms": utt.get("start_ms"),
                "end_ms": utt.get("end_ms"),
            },
            "expected_output": expected_output,
        }
        
        items.append(item)
    
    return {
        "name": f"long_conversation_{audio_stem}",
        "type": "long_conversation",
        "source_audio": audio_name,
        "language": language,
        "seller_speaker_id": seller_speaker_id,
        "total_utterances": len(utterances),
        "items": items,
    }


def load_speaker_map(base_dir: Path) -> dict:
    """Load speaker map from base directory if it exists."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a long conversation dataset for seller response testing."
    )
    parser.add_argument(
        "--conversation-path",
        required=True,
        help="Path to a processed conversation JSON file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for the dataset JSON file. Defaults to same directory as input with '_dataset' suffix.",
    )
    parser.add_argument(
        "--speaker-map",
        help="Path to speaker_map.json file. If not provided, will look in parent directory.",
    )
    parser.add_argument(
        "--seller-speaker-id",
        type=int,
        choices=[1, 2],
        help="Explicitly set which speaker_id is the seller (1 or 2). Overrides speaker map.",
    )
    parser.add_argument(
        "--push-to-langfuse",
        action="store_true",
        default=False,
        help="Upload generated dataset items to Langfuse.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name for the Langfuse dataset. Required if --push-to-langfuse is set.",
    )
    return parser.parse_args()


def create_langfuse_dataset(langfuse_client, dataset_name: str, language: str = "Dansk") -> None:
    """Create a Langfuse dataset if it doesn't exist."""
    if langfuse_client is None:
        return
    try:
        import time
        time.sleep(1)
        langfuse_client.create_dataset(
            name=dataset_name,
            description="Long conversation dataset for seller response testing.",
            metadata={"language": language, "type": "long_conversation"},
        )
        print(f"Created Langfuse dataset: {dataset_name}")
    except Exception as exc:
        # Dataset might already exist; log and continue.
        print(f"[langfuse] create_dataset skipped (may already exist): {exc}")


def push_items_to_langfuse(
    langfuse_client,
    dataset_name: str,
    items: List[dict],
    conversation_name: str,
) -> int:
    """Push dataset items to Langfuse."""
    if langfuse_client is None or not items:
        return 0

    import time
    created = 0
    for idx, item in enumerate(items):
        metadata = dict(item.get("metadata", {}))
        metadata["conversation"] = conversation_name
        time.sleep(0.5)  # Rate limiting
        try:
            langfuse_client.create_dataset_item(
                dataset_name=dataset_name,
                input=item["input"],
                metadata=metadata,
                expected_output=item["expected_output"],
            )
            created += 1
            if (idx + 1) % 10 == 0:
                print(f"  -> Pushed {idx + 1}/{len(items)} items...")
        except Exception as exc:
            print(f"[langfuse] Failed to push item {idx}: {exc}")
    return created


def main() -> None:
    args = parse_args()
    
    # Validate Langfuse arguments
    if args.push_to_langfuse and not args.dataset_name:
        raise ValueError("--dataset-name is required when using --push-to-langfuse")
    
    conversation_path = Path(args.conversation_path).expanduser().resolve()
    if not conversation_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {conversation_path}")
    
    # Initialize Langfuse if needed
    langfuse_client = None
    if args.push_to_langfuse:
        langfuse_client = init_langfuse(True)
        if langfuse_client is None:
            raise ValueError("Failed to initialize Langfuse client. Check your credentials.")
    
    # Load speaker map
    speaker_map = {}
    if args.speaker_map:
        speaker_map_path = Path(args.speaker_map).expanduser().resolve()
        if speaker_map_path.exists():
            speaker_map = load_speaker_map(speaker_map_path.parent)
    else:
        # Try to find speaker_map.json in parent directories
        for parent in conversation_path.parents:
            speaker_map_path = parent / "speaker_map.json"
            if speaker_map_path.exists():
                speaker_map = load_speaker_map(parent)
                break
    
    # Load conversation
    print(f"Loading conversation: {conversation_path.name}")
    conversation_data = load_conversation(conversation_path)
    
    # Override seller speaker ID if explicitly provided
    if args.seller_speaker_id:
        audio_stem = conversation_data.get("audio_stem", conversation_path.stem)
        # Inject into speaker_map to override
        speaker_map[audio_stem] = (args.seller_speaker_id == 1)
    
    # Generate dataset
    dataset = generate_long_conversation_dataset(conversation_data, speaker_map)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = conversation_path.parent / f"{conversation_path.stem}_long_conv_dataset.json"
    
    # Save dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Generated dataset with {len(dataset['items'])} seller turn items")
    print(f"Seller speaker ID: {dataset.get('seller_speaker_id', 'unknown')}")
    print(f"Total utterances in conversation: {dataset.get('total_utterances', 0)}")
    print(f"Saved to: {output_path}")
    
    # Push to Langfuse if requested
    if args.push_to_langfuse and langfuse_client:
        print(f"\nPushing to Langfuse dataset: {args.dataset_name}")
        create_langfuse_dataset(langfuse_client, args.dataset_name, dataset.get("language", "Dansk"))
        
        created = push_items_to_langfuse(
            langfuse_client=langfuse_client,
            dataset_name=args.dataset_name,
            items=dataset["items"],
            conversation_name=conversation_path.name,
        )
        print(f"Pushed {created} items to Langfuse dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()
