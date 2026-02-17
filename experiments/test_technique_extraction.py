#!/usr/bin/env python3
"""
Test the technique extraction prompt against the 3 sample transcripts.

Usage:
    cd experiments
    python test_technique_extraction.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
os.chdir(PROJECT_ROOT)

from services.llm_client import GeminiChatCompletionClient

PROMPT_DIR = PROJECT_ROOT / "prompts" / "technique_extraction"
PROMPT_FILE = PROMPT_DIR / "v1_prompt.txt"
CONFIG_FILE = PROMPT_DIR / "v1_config.json"

TRANSCRIPTS_DIR = PROJECT_ROOT / "customer_data" / "hello_sales" / "elg_b2c" / "analysis_results"
OUTPUT_DIR = PROJECT_ROOT / "experiments"


def extract_transcript(file_path: Path) -> str:
    """Extract the transcript string from an analysis result file."""
    raw = file_path.read_text(encoding="utf-8")
    data = json.loads(raw.split("\n```json")[0])
    return data["transcript"]


def main() -> None:
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    config_data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    client = GeminiChatCompletionClient(
        default_model="gemini-3-flash-preview",
        thinking_level="low",
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {"schema": config_data["json_schema"]},
    }

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("analysis_result_*.txt"))
    print(f"Found {len(transcript_files)} transcripts\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for tf in transcript_files:
        name = tf.stem
        print(f"{'=' * 60}")
        print(f"Processing: {name}")
        print(f"{'=' * 60}")

        transcript = extract_transcript(tf)
        print(f"  Transcript length: {len(transcript)} chars")

        messages = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": f"## Transskript\n\n{transcript}"},
        ]

        try:
            result = client.generate(
                messages=messages,
                response_format=response_format,
            )
            parsed = json.loads(result)

            # Quick validation summary
            summary = parsed.get("call_summary", {})
            techniques = parsed.get("technique_instances", [])
            print(f"  Outcome: {summary.get('call_outcome')}")
            print(f"  Trajectory: {summary.get('engagement_trajectory')}")
            print(f"  Phases: {summary.get('phases_reached')}")
            print(f"  Phase lost: {summary.get('phase_where_lost')}")
            print(f"  Techniques extracted: {len(techniques)}")

            for i, t in enumerate(techniques):
                eb = t.get("engagement_before")
                ea = t.get("engagement_after")
                print(f"    [{i+1}] {t['primary_category']}/{t['technique_subtype']} "
                      f"({t['phase_context']}) {eb}->{ea} "
                      f"progression={t['led_to_progression']}")
                quote_preview = t["seller_quote"][:80]
                print(f"        quote: \"{quote_preview}...\"")

            # Save full output
            out_path = OUTPUT_DIR / f"technique_extraction_{name}_{timestamp}.json"
            out_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Saved: {out_path.name}\n")

        except Exception as exc:
            print(f"  ERROR: {exc}\n")


if __name__ == "__main__":
    main()
