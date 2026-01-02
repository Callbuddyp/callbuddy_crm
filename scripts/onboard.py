#!/usr/bin/env python3
"""
Onboarding script for new campaigns.

This script orchestrates the complete onboarding process:
1. Validates and loads unified campaign configuration
2. Generates campaign information from raw materials (optional)
3. Creates action prompts in Langfuse (optional)
4. Generates datasets from audio files (optional)

Usage:
  python onboard.py --customer-data-dir customer_data/hello_sales/elg_b2c
  
Options:
  --skip-dataset        Skip dataset generation (useful when no audio files)
  --skip-prompts        Skip prompt creation in Langfuse
  --skip-campaign-info  Skip campaign info generation
  --dry-run             Show what would be done without executing
  --push-to-langfuse    Push generated datasets to Langfuse
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from services.campaign_info_generator import generate_campaign_info
from services.prompt_generator import create_campaign_prompts
from utils import load_env_value



@dataclass
class CampaignConfig:
    """Campaign configuration section."""
    id: str
    name: str
    description: str = ""
    language: str = "Dansk"


@dataclass
class DatasetConfig:
    """Dataset configuration section."""
    name: str
    action_types: List[str] = field(default_factory=list)


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    speech_minimum_duration_ms: int = 2000
    silence_before_ai_ms: int = 500


@dataclass
class ActionPromptConfig:
    """Configuration for a single action prompt.
    
    Supports two structures:
    - 'standard': Uses component arrays for system/user prompts
    - 'custom': Uses inline prompt_content with optional features
    """
    langfuse_name: str
    enabled: bool = True
    structure: str = "standard"  # 'standard' or 'custom'
    
    # Frontend configuration (use_reasoning, use_internet, output_type, etc.)
    frontend_config: Dict[str, any] = field(default_factory=dict)
    
    # For 'standard' structure
    components: Dict[str, List[str]] = field(default_factory=dict)
    
    # For 'custom' structure
    prompt_content: Optional[str] = None
    
    # Legacy field for backward compatibility
    action_prompt: str = ""


@dataclass
class PromptsDefaultsConfig:
    """Default prompt component configuration."""
    base_system_prompt: str = "salescall_system1"
    output_format_prompt: str = "output_markdown_suggestions"


@dataclass
class PromptsConfig:
    """Prompts configuration section."""
    generate_campaign_info: bool = False
    defaults: PromptsDefaultsConfig = field(default_factory=PromptsDefaultsConfig)
    actions: Dict[str, ActionPromptConfig] = field(default_factory=dict)
    
    # Legacy fields for backward compatibility
    base_system_prompt: str = "salescall_system1"
    output_format_prompt: str = "output_markdown_suggestions"


@dataclass
class UnifiedConfig:
    """Complete unified configuration for campaign onboarding."""
    campaign: CampaignConfig
    dataset: DatasetConfig
    vad: VADConfig
    prompts: PromptsConfig
    campaign_info_files: List[str] = field(default_factory=list)
    
    # Store raw prompts data for prompt_generator access
    _raw_prompts_data: dict = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedConfig":
        """Create UnifiedConfig from dictionary (loaded from JSON)."""
        # Parse campaign section
        campaign_data = data.get("campaign", {})
        campaign = CampaignConfig(
            id=campaign_data.get("id", ""),
            name=campaign_data.get("name", ""),
            description=campaign_data.get("description", ""),
            language=campaign_data.get("language", "Dansk"),
        )
        
        # Parse dataset section
        dataset_data = data.get("dataset", {})
        dataset = DatasetConfig(
            name=dataset_data.get("name", campaign.id),
            action_types=dataset_data.get("action_types", []),
        )
        
        # Parse VAD section
        vad_data = data.get("vad", {})
        vad = VADConfig(
            speech_minimum_duration_ms=vad_data.get("speech_minimum_duration_ms", 2000),
            silence_before_ai_ms=vad_data.get("silence_before_ai_ms", 500),
        )
        
        # Parse prompts section
        prompts_data = data.get("prompts", {})
        
        # Parse defaults
        defaults_data = prompts_data.get("defaults", {})
        defaults = PromptsDefaultsConfig(
            base_system_prompt=defaults_data.get("base_system_prompt", 
                prompts_data.get("base_system_prompt", "salescall_system1")),
            output_format_prompt=defaults_data.get("output_format_prompt",
                prompts_data.get("output_format_prompt", "output_markdown_suggestions")),
        )
        
        # Parse actions with flexible structure
        actions = {}
        for action_type, action_data in prompts_data.get("actions", {}).items():
            structure = action_data.get("structure", "standard")
            
            actions[action_type] = ActionPromptConfig(
                langfuse_name=action_data.get("langfuse_name", f"{campaign.id}_{action_type}"),
                enabled=action_data.get("enabled", True),
                structure=structure,
                frontend_config=action_data.get("config", {}),
                components=action_data.get("components", {}),
                prompt_content=action_data.get("prompt_content"),
                action_prompt=action_data.get("action_prompt", f"{action_type}_prompt"),
            )
        
        prompts = PromptsConfig(
            generate_campaign_info=prompts_data.get("generate_campaign_info", False),
            defaults=defaults,
            actions=actions,
            base_system_prompt=prompts_data.get("base_system_prompt", "salescall_system1"),
            output_format_prompt=prompts_data.get("output_format_prompt", "output_markdown_suggestions"),
        )
        
        return cls(
            campaign=campaign,
            dataset=dataset,
            vad=vad,
            prompts=prompts,
            campaign_info_files=data.get("campaign_info_files", []),
            _raw_prompts_data=prompts_data,
        )
    
    def validate(self) -> List[str]:
        """Validate the configuration. Returns list of errors (empty if valid)."""
        errors = []
        
        if not self.campaign.id:
            errors.append("campaign.id is required")
        if not self.campaign.name:
            errors.append("campaign.name is required")
        
        return errors
    
    def get_action_types(self) -> List[str]:
        """Get action types from enabled prompts.actions."""
        return [
            action_type for action_type, action_config 
            in self.prompts.actions.items() 
            if action_config.enabled
        ]


def load_unified_config(base_dir: Path) -> UnifiedConfig:
    """Load and validate the unified configuration from config.json."""
    config_path = base_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    config = UnifiedConfig.from_dict(data)
    
    errors = config.validate()
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return config


def check_audio_files(base_dir: Path) -> bool:
    """Check if audio files exist in the conversations directory."""
    audio_dir = base_dir / "conversations"
    if not audio_dir.exists():
        return False
    
    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in audio_extensions:
            return True
    return False


def check_campaign_info_files(base_dir: Path, files: List[str]) -> List[Path]:
    """Check which campaign info files exist and return their paths."""
    campaign_info_dir = base_dir / "campaign_info"
    existing_files = []
    
    if not campaign_info_dir.exists():
        return existing_files
    
    for filename in files:
        path = campaign_info_dir / filename
        if path.exists():
            existing_files.append(path)
    
    return existing_files


def _str_to_bool(value: str | bool) -> bool:
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value (true/false).")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Onboard a new campaign with dataset generation, prompt creation, and campaign info."
    )
    parser.add_argument(
        "--customer-data-dir",
        required=True,
        help="Path to customer data directory containing config.json",
    )
    parser.add_argument(
        "--skip-dataset",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Skip dataset generation from audio files",
    )
    parser.add_argument(
        "--skip-prompts",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Skip prompt creation in Langfuse",
    )
    parser.add_argument(
        "--skip-campaign-info",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Skip campaign info generation",
    )
    parser.add_argument(
        "--push-to-langfuse",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Push generated datasets to Langfuse",
    )
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--preview",
        nargs="?",
        const=True,
        default=False,
        type=_str_to_bool,
        help="Preview LLM-generated campaign info before uploading",
    )
    return parser.parse_args()


def run_dataset_generation(base_dir: Path, config: UnifiedConfig, push_to_langfuse: bool, dry_run: bool) -> None:
    """Run dataset generation using the existing gen_conv_datasets logic."""
    if dry_run:
        print("  [DRY RUN] Would generate datasets from audio files")
        return
    
    # Import here to avoid circular imports
    from gen_conv_datasets import main as gen_datasets_main
    
    # Build the arguments for gen_conv_datasets
    import sys
    original_argv = sys.argv
    sys.argv = [
        "gen_conv_datasets.py",
        "--customer-data-dir", str(base_dir),
    ]
    if push_to_langfuse:
        sys.argv.append("--push-to-langfuse")
    
    try:
        gen_datasets_main()
    finally:
        sys.argv = original_argv


def main() -> None:
    """Main entry point for the onboarding script."""
    args = parse_args()
    base_dir = Path(args.customer_data_dir).expanduser().resolve()
    
    print("=" * 60)
    print("CAMPAIGN ONBOARDING SCRIPT")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No changes will be made\n")
    
    # Step 1: Load and validate configuration
    print("\n[1/4] Loading configuration...")
    try:
        config = load_unified_config(base_dir)
        print(f"  ✓ Campaign: {config.campaign.name} ({config.campaign.id})")
        print(f"  ✓ Language: {config.campaign.language}")
        print(f"  ✓ Action types: {config.get_action_types()}")
    except Exception as e:
        print(f"  ✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Step 2: Generate campaign info (if enabled)
    print("\n[2/4] Campaign information generation...")
    if args.skip_campaign_info:
        print("  → Skipped (--skip-campaign-info)")
    elif not config.prompts.generate_campaign_info:
        print("  → Disabled in config (prompts.generate_campaign_info = false)")
    else:
        campaign_info_files = check_campaign_info_files(base_dir, config.campaign_info_files)
        if not campaign_info_files:
            print("  → No campaign info files found, skipping")
        else:
            print(f"  → Found {len(campaign_info_files)} campaign info files")
            if args.dry_run:
                print("  [DRY RUN] Would generate campaign info from files")
            else:
                try:
                    campaign_info = generate_campaign_info(
                        config=config,
                        files=campaign_info_files,
                        output_dir=base_dir / "campaign_info",
                    )
                    print(f"  ✓ Generated campaign info")
                    
                    # Show preview if requested
                    if args.preview:
                        print(f"\n{'='*60}")
                        print("CAMPAIGN INFO PREVIEW")
                        print(f"{'='*60}")
                        if len(campaign_info) > 2000:
                            print(campaign_info[:2000])
                            print(f"\n... ({len(campaign_info) - 2000} more chars)")
                        else:
                            print(campaign_info)
                        print(f"{'='*60}")
                        
                        choice = input("\n[a]ccept and continue / [q]uit: ").strip().lower()
                        if choice in ("q", "quit", "abort", "exit"):
                            print("  ✗ Aborted by user")
                            sys.exit(0)
                    
                    # Upload to Langfuse so prompts can reference it
                    print(f"  → Uploading campaign info to Langfuse...")
                    from services.prompt_generator import upload_campaign_info_prompt
                    prompt_name = upload_campaign_info_prompt(config, campaign_info)
                    print(f"  ✓ Uploaded as '{prompt_name}' to Langfuse")
                    
                except Exception as e:
                    print(f"  ✗ Failed to generate/upload campaign info: {e}")
                    print("  ✗ Stopping - campaign info is required for prompts")
                    sys.exit(1)
    
    # Step 3: Create prompts in Langfuse (if enabled)
    print("\n[3/4] Prompt creation in Langfuse...")
    if args.skip_prompts:
        print("  → Skipped (--skip-prompts)")
    else:
        enabled_actions = [
            (action_type, action_config)
            for action_type, action_config in config.prompts.actions.items()
            if action_config.enabled
        ]
        if not enabled_actions:
            print("  → No action prompts enabled in config")
        else:
            print(f"  → Creating prompts for: {[a[0] for a in enabled_actions]}")
            if args.dry_run:
                print("  [DRY RUN] Would create prompts in Langfuse")
            else:
                try:
                    create_campaign_prompts(config=config)
                    print(f"  ✓ Created {len(enabled_actions)} prompts in Langfuse")
                except Exception as e:
                    print(f"  ✗ Failed to create prompts: {e}")
                    sys.exit(1)
    
    # Step 4: Generate datasets (if enabled and audio files exist)
    print("\n[4/4] Dataset generation from audio files...")
    if args.skip_dataset:
        print("  → Skipped (--skip-dataset)")
    else:
        has_audio = check_audio_files(base_dir)
        if not has_audio:
            print("  → No audio files found in conversations/, skipping")
        else:
            print("  → Audio files found, generating datasets...")
            try:
                run_dataset_generation(base_dir, config, args.push_to_langfuse, args.dry_run)
                print("  ✓ Dataset generation complete")
            except Exception as e:
                print(f"  ✗ Failed to generate datasets: {e}")
    
    print("\n" + "=" * 60)
    print("ONBOARDING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
