#!/usr/bin/env python3
"""
Interactive CLI for generating competitor intelligence.

Thin wrapper around CompetitorFetcherService — collects user input,
displays results, and handles the human checkpoint between phases.

Usage:
    cd scripts
    python run_competitor_fetcher.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
os.chdir(_PROJECT_ROOT)

from services.competitor_fetcher import (
    CompetitorFetcherService,
    format_research_results_md,
)
from services.langfuse import init_langfuse, _with_rate_limit_backoff
from services.supabase_client import SupabaseService
from utils import load_env_value


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


def prompt_string(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    raw = input(f"{C.BOLD}{question}{suffix}: {C.RESET}").strip()
    return raw if raw else default


def prompt_confirm(question: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{C.BOLD}{question} ({hint}): {C.RESET}").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def collect_multiline(label: str) -> str:
    print(f"{C.BOLD}{label} (empty line to finish):{C.RESET}")
    lines = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_competitors(competitors: list[dict]) -> None:
    """Print the Phase 1 competitor list with details."""
    for i, comp in enumerate(competitors, 1):
        name = comp.get("name", "?")
        aliases = comp.get("aliases", [])
        alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
        urls = comp.get("discovered_urls", [])
        focus = comp.get("focus_areas", "")
        print(f"  {C.CYAN}{i}{C.RESET}) {C.BOLD}{name}{C.RESET}{alias_str}")
        if urls:
            short_urls = [u.split("//")[1] if "//" in u else u for u in urls[:3]]
            print(f"     URLs: {', '.join(short_urls)}")
        if focus:
            print(f"     Focus: {C.DIM}{focus[:120]}{'...' if len(focus) > 120 else ''}{C.RESET}")
        print()


def edit_competitor_list(competitors: list[dict]) -> list[dict]:
    """Interactive editing of the competitor list. Returns modified list."""
    if not prompt_confirm("Edit competitor list?", default=False):
        return competitors

    remove_input = prompt_string("Remove (comma-separated numbers, or empty)")
    if remove_input:
        indices = {int(x.strip()) - 1 for x in remove_input.split(",") if x.strip().isdigit()}
        competitors = [c for i, c in enumerate(competitors) if i not in indices]
        info(f"Removed {len(indices)} competitor(s). {len(competitors)} remaining.")

    while True:
        add_name = prompt_string("Add competitor name (empty to finish)")
        if not add_name:
            break
        competitors.append({
            "name": add_name, "aliases": [], "rationale": "Manually added",
            "discovered_urls": [], "focus_areas": "",
        })
        info(f"Added '{add_name}'")

    return competitors


# ---------------------------------------------------------------------------
# Dump helpers
# ---------------------------------------------------------------------------

def _save_raw(path: Path, data) -> None:
    """Save raw output to disk, handling pydantic models and other types."""
    try:
        if hasattr(data, "model_dump"):
            serializable = data.model_dump()
        elif isinstance(data, str):
            serializable = json.loads(data)
        else:
            serializable = data
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
        success(f"Raw output saved to {path}")
    except Exception as exc:
        warn(f"Could not save raw output: {exc}")
        # Last resort: dump repr
        with path.open("w", encoding="utf-8") as f:
            f.write(repr(data))
        warn(f"Saved repr fallback to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    header("Competitor Intelligence Fetcher")

    # --- Save mode ---
    save_mode = prompt_choice(
        "Where should results be saved?",
        ["1 - Local (JSON files)", "2 - Supabase", "3 - Local Supabase"],
    )
    local_mode = save_mode.startswith("1")
    local_supabase = save_mode.startswith("3")
    if local_mode:
        info("Running in local mode (no Supabase)")
    if local_supabase:
        info("Using local Supabase (127.0.0.1:54321)")

    # --- Campaign ID ---
    campaign_id = prompt_string("Campaign ID")
    if not campaign_id:
        error("Campaign ID is required.")
        sys.exit(1)

    # --- Campaign info source ---
    info_source = prompt_choice(
        "Where to load campaign info from?",
        ["1 - Langfuse prompt", "2 - Local file"],
    )

    if info_source.startswith("2"):
        info_file = prompt_string("Path to campaign info file")
        info_path = Path(info_file).expanduser().resolve()
        if not info_path.exists():
            error(f"File not found: {info_path}")
            sys.exit(1)
        campaign_info = info_path.read_text(encoding="utf-8")
        success(f"Loaded campaign info from {info_path} ({len(campaign_info)} chars)")
    else:
        prompt_name = prompt_string("Langfuse prompt name", default=f"{campaign_id}_info")

        info("Initializing Langfuse...")
        langfuse_client = init_langfuse(push_to_langfuse=True)
        if not langfuse_client:
            error("Could not initialize Langfuse.")
            sys.exit(1)

        info(f"Fetching '{prompt_name}'...")
        try:
            prompt_obj = _with_rate_limit_backoff(
                lambda: langfuse_client.get_prompt(name=prompt_name, type="text")
            )
            campaign_info = prompt_obj.prompt
            success(f"Fetched campaign info ({len(campaign_info)} chars)")
        except Exception as exc:
            error(f"Failed to fetch prompt: {exc}")
            sys.exit(1)

    custom_messages = collect_multiline("Custom messages (optional)")
    source_urls = collect_multiline("Source URLs (optional)")

    # --- Pipeline version ---
    pipeline = prompt_choice(
        "Which pipeline?",
        ["1 - v2 (Researcher + Evaluator, ~30-60s)", "2 - v1 (Discovery + Parallel AI, 5-25 min)"],
    )
    use_v2 = pipeline.startswith("1")

    # --- Output directory (local mode) ---
    if local_mode:
        default_out = str(_PROJECT_ROOT / "customer_data" / campaign_id)
        out_dir = Path(prompt_string("Output directory", default=default_out))
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize service ---
    gemini_key = load_env_value("GEMINI_API_KEY")
    if not gemini_key:
        error("GEMINI_API_KEY is required.")
        sys.exit(1)

    parallel_key = ""
    if not use_v2:
        parallel_key = load_env_value("PARALLEL_AI_API_KEY")
        if not parallel_key:
            error("PARALLEL_AI_API_KEY is required for v1 pipeline.")
            sys.exit(1)

    supabase_service = None
    if local_supabase:
        _LOCAL_SB_URL = "http://127.0.0.1:54321"
        _LOCAL_SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
        info(f"Connecting to {_LOCAL_SB_URL}...")
        supabase_service = SupabaseService(_LOCAL_SB_URL, _LOCAL_SB_KEY)
        success("Connected to local Supabase")
    elif not local_mode:
        sb_url = load_env_value("SUPABASE_URL")
        sb_key = load_env_value("SUPABASE_SERVICE_ROLE_KEY")
        if sb_url and sb_key:
            supabase_service = SupabaseService(sb_url, sb_key)
        else:
            warn("Supabase credentials not found. Results will be saved locally.")
            local_mode = True
            default_out = str(_PROJECT_ROOT / "customer_data" / campaign_id)
            out_dir = Path(prompt_string("Output directory", default=default_out))
            out_dir.mkdir(parents=True, exist_ok=True)

    service = CompetitorFetcherService(
        gemini_api_key=gemini_key,
        parallel_api_key=parallel_key,
        supabase_service=supabase_service,
    )

    # --- v2 pipeline ---
    if use_v2:
        # --- v2: Researcher ---
        header("Phase 1: Researching Competitors (Gemini + Search)")
        t0 = time.perf_counter()
        try:
            competitors = service.research_competitors_v2(
                campaign_info=campaign_info,
                custom_messages=custom_messages,
                source_urls=source_urls,
            )
        except Exception as exc:
            error(f"Research failed: {exc}")
            sys.exit(1)
        success(f"Found {len(competitors)} competitors ({time.perf_counter() - t0:.1f}s)")

        # Save researcher output
        dump_dir = _PROJECT_ROOT / "customer_data" / campaign_id
        dump_dir.mkdir(parents=True, exist_ok=True)
        researcher_path = dump_dir / "competitor_researcher_output.json"
        _save_raw(researcher_path, competitors)

        # Display competitor names
        print()
        for i, comp in enumerate(competitors, 1):
            name = comp.get("name", "?")
            aliases = comp.get("aliases", [])
            alias_str = f" ({', '.join(aliases)})" if aliases else ""
            print(f"  {C.CYAN}{i}{C.RESET}) {C.BOLD}{name}{C.RESET}{alias_str}")
        print()

        # --- v2: Evaluator ---
        skip_eval = not prompt_confirm("Run evaluator to improve quality?")
        if not skip_eval:
            header("Phase 2: Evaluating & Improving (Gemini + Search)")
            t0 = time.perf_counter()
            try:
                competitors = service.evaluate_competitors(
                    campaign_info=campaign_info,
                    researcher_output=competitors,
                )
            except Exception as exc:
                error(f"Evaluation failed: {exc}")
                warn("Falling back to researcher output.")
            else:
                success(f"Evaluation complete ({time.perf_counter() - t0:.1f}s)")

        # Save final output
        final_path = dump_dir / "competitor_research_v2.json"
        _save_raw(final_path, competitors)

        # --- Display ---
        header("Results")
        from services.competitor_fetcher import render_competitor_description
        for comp in competitors:
            print(render_competitor_description(comp))
            print(f"\n{'─' * 40}\n")
        info(f"{len(competitors)} competitors researched")

        # --- Store ---
        if local_mode:
            success(f"Results saved to {final_path}")
        elif supabase_service:
            target = "local Supabase" if local_supabase else "Supabase"
            if prompt_confirm(f"Save to {target}?"):
                count = service.store_results_v2(campaign_id, competitors)
                success(f"Stored {count} competitors for '{campaign_id}' in {target}")

        header("Done!")
        return

    # --- Phase 1: Discovery ---
    header("Phase 1: Discovering Competitors (Gemini + Search)")
    t0 = time.perf_counter()
    try:
        competitors = service.discover_competitors(
            campaign_info=campaign_info,
            custom_messages=custom_messages,
            source_urls=source_urls,
        )
    except Exception as exc:
        error(f"Discovery failed: {exc}")
        sys.exit(1)
    success(f"Found {len(competitors)} competitors ({time.perf_counter() - t0:.1f}s)")

    print()
    display_competitors(competitors)
    competitors = edit_competitor_list(competitors)

    if not competitors:
        error("No competitors to research.")
        sys.exit(1)

    # Dump directory for saving intermediate results
    dump_dir = _PROJECT_ROOT / "customer_data" / campaign_id
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Save Phase 1 discovery
    p1_path = dump_dir / "competitor_discovery.json"
    with p1_path.open("w", encoding="utf-8") as f:
        json.dump(competitors, f, ensure_ascii=False, indent=2)
    success(f"Phase 1 saved to {p1_path}")

    # --- Phase 2: Deep Research ---
    # Check for existing research output to reload
    raw_dump_path = dump_dir / "competitor_research_raw.json"
    research_output = None

    if raw_dump_path.exists():
        reload = prompt_choice(
            f"Found previous research output at {raw_dump_path}. What to do?",
            ["1 - Load it (skip Phase 2)", "2 - Re-run Phase 2 (overwrite)"],
        )
        if reload.startswith("1"):
            with raw_dump_path.open("r", encoding="utf-8") as f:
                research_output = json.load(f)
            success(f"Loaded previous research output ({len(research_output.get('competitors', []))} competitors)")

    if research_output is None:
        processor = prompt_choice(
            "Processor tier?",
            ["1 - pro ($0.10, 5-15 min)", "2 - ultra ($0.30, 10-25 min)"],
        )
        processor = "ultra" if processor.startswith("2") else "pro"

        header(f"Phase 2: Deep Research (Parallel AI, {processor})")
        info("Submitting research task (this may take 5-25 minutes)...")
        t0 = time.perf_counter()
        try:
            research_output = service.research_competitors(
                campaign_info=campaign_info,
                competitors=competitors,
                custom_messages=custom_messages,
                processor=processor,
            )
        except Exception as exc:
            error(f"Research failed: {exc}")
            sys.exit(1)
        elapsed = time.perf_counter() - t0
        success(f"Research complete ({elapsed / 60:.1f} min)")

        # Always save raw output immediately (before any processing)
        _save_raw(raw_dump_path, research_output)

    # --- Display results ---
    header("Results")
    try:
        print(format_research_results_md(research_output))
        info(f"{len(research_output.get('competitors', []))} competitors researched")
    except Exception as exc:
        error(f"Failed to format results: {exc}")
        info(f"Raw output saved at {raw_dump_path} for debugging")
        sys.exit(1)

    # --- Phase 3: Save ---
    if local_mode:
        result_path = out_dir / "competitor_research.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(research_output, f, ensure_ascii=False, indent=2)
        success(f"Results saved to {result_path}")
    elif supabase_service:
        target = "local Supabase" if local_supabase else "Supabase"
        if prompt_confirm(f"Save to {target}?"):
            count = service.store_results(campaign_id, research_output, competitors)
            success(f"Stored {count} competitors for '{campaign_id}' in {target}")
    else:
        local_path = dump_dir / "competitor_research.json"
        if prompt_confirm(f"Save locally to {local_path}?"):
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(research_output, f, ensure_ascii=False, indent=2)
            success(f"Saved to {local_path}")

    header("Done!")


if __name__ == "__main__":
    main()
