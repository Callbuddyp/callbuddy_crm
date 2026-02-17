#!/usr/bin/env python3
"""
Onboarding script for new campaigns.

This script orchestrates the complete onboarding process:
1. Validates and loads unified campaign configuration
2. Syncs Firm, Campaign, and Users to Supabase (if configured)
3. Generates campaign information from raw materials (optional)
4. Creates action prompts in Langfuse (optional)
5. Generates datasets from audio files (optional)

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
import os
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from services.campaign_info_generator import generate_campaign_info
from services.prompt_generator import create_campaign_prompts
from services.supabase_client import SupabaseService
from utils import load_env_value



@dataclass
class FirmConfig:
    """Firm configuration (loaded from firm_config.json in parent directory)."""
    name: str

@dataclass
class AdminUserConfig:
    """Admin user configuration."""
    email: str
    name: Optional[str] = None

@dataclass
class CampaignConfig:
    """Campaign configuration section."""
    id: str
    name: str
    description: str = ""
    language: str = "Dansk"
    status: str = "draft"  # Added for Supabase sync


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
    
    # Supabase / Firm config (Optional)
    firm: Optional[FirmConfig] = None
    api_keys: Dict[str, str] = field(default_factory=dict)
    admin_users: List[AdminUserConfig] = field(default_factory=list)
    
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
            status=campaign_data.get("status", "draft"),
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
        
        # Note: Firm config is now loaded separately from firm_config.json
        # Legacy support: still parse firm from campaign config if present
        firm = None
        if "firm" in data:
            firm = FirmConfig(name=data["firm"].get("name", ""))
            
        # Parse Admin Users
        admin_users = []
        for au in data.get("admin_users", []):
            admin_users.append(AdminUserConfig(
                email=au.get("email", ""),
                name=au.get("name")
            ))
            
        return cls(
            campaign=campaign,
            dataset=dataset,
            vad=vad,
            prompts=prompts,
            firm=firm,
            api_keys=data.get("api_keys", {}),
            admin_users=admin_users,
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


def load_firm_config(base_dir: Path, campaign_config: UnifiedConfig) -> Optional[FirmConfig]:
    """Load firm configuration from firm_config.json in parent directory.
    
    Looks for firm_config.json in the parent directory (e.g., customer_data/daica/firm_config.json
    for a campaign at customer_data/daica/telemore/).
    
    Falls back to legacy firm config in campaign's config.json if firm_config.json not found.
    
    Args:
        base_dir: Campaign directory path
        campaign_config: Already loaded campaign configuration (for legacy fallback)
    
    Returns:
        FirmConfig if found, None otherwise
    """
    # Look for firm_config.json in parent directory
    firm_config_path = base_dir.parent / "firm_config.json"
    
    if firm_config_path.exists():
        try:
            with firm_config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            firm_name = data.get("name", "")
            if firm_name:
                return FirmConfig(name=firm_name)
            else:
                print(f"  ⚠️ firm_config.json found but 'name' is missing")
        except Exception as e:
            print(f"  ⚠️ Error loading firm_config.json: {e}")
    
    # Fallback to legacy firm in campaign config
    if campaign_config.firm:
        return campaign_config.firm
    
    return None


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


def get_campaign_info_files(base_dir: Path) -> List[Path]:
    """Get all files in the campaign_info directory."""
    campaign_info_dir = base_dir / "campaign_info"
    
    if not campaign_info_dir.exists():
        return []
        
    return [
        p for p in campaign_info_dir.glob("*") 
        if p.is_file() and not p.name.startswith(".")
    ]


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
    parser.add_argument(
        "--skip-supabase",
        action="store_true",
        help="Skip Supabase syncing",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Supabase instance (http://127.0.0.1:54321)",
    )
    parser.add_argument(
        "--create-users-only",
        action="store_true",
        help="Only create/link users to campaign (skips prompts, campaign info, datasets, and prompt templates)",
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

def run_supabase_onboarding(config: UnifiedConfig, firm_config: Optional[FirmConfig], base_dir: Path, dry_run: bool, use_local: bool = False, users_only: bool = False) -> None:
    """Run Supabase onboarding (Firm, Campaign, Users).
    
    Args:
        config: Campaign configuration
        firm_config: Firm configuration (loaded from firm_config.json or legacy)
        base_dir: Campaign directory path
        dry_run: If True, only show what would be done
        use_local: If True, use local Supabase instance
        users_only: If True, only process users (skip prompt templates)
    """

    if dry_run:
        print("  [DRY RUN] Would sync Firm, Campaign, and Users to Supabase")
        return

    print("  → Connecting to Supabase...")
    svc = SupabaseService(supabase_url, supabase_key)
    
    # 1. Firm
    firm_id = ""
    if firm_config:
        print(f"  → Processing Firm: {firm_config.name}")
        firm_id = svc.get_or_create_firm(name=firm_config.name)

        # Keys (from campaign config)
        if config.api_keys:
            for provider, key in config.api_keys.items():
                if key:
                    svc.set_firm_api_key(firm_id, key, provider)
    else:
        print("  ! No Firm configuration found, skipping firm creation.")
        # Campaigns require firm_id, so we stop here
        return

    # 2. Campaign
    print(f"  → Processing Campaign: {config.campaign.name}")
    campaign_id = svc.get_or_create_campaign(firm_id, config.campaign.name, config.campaign.status)

    # 3. Prompt Templates (skip if users_only)
    if not users_only:
        # Each action in prompts.actions corresponds to a template? 
        # onboard.ts had explicit prompt templates in config.
        # We can infer from `config.prompts.actions`.
        found_templates = 0
        for action_type, action_config in config.prompts.actions.items():
            if action_config.enabled:
                 # name needs to match what the app expects.
                 # In TS: "name": "elg_b2c_text", "action": "text"
                 # In Python config: keys are action_type, values have langfuse_name.
                 # We should probably use `langfuse_name` as the template name? Or just the action type?
                 # onboard.ts used `prompt.name` which mapped to `prompt_template_name`.
                 # Check TS config: "name": "elg_b2c_text", "action": "text"
                 # In Python config: action_type="text", langfuse_name="elg_b2c_text"
                 # So we use langfuse_name.
                 svc.upsert_prompt_template(campaign_id, action_config.langfuse_name, action_type)
                 found_templates += 1
        if found_templates > 0:
            print(f"  ✓ Processed {found_templates} prompt templates")
    else:
        print("  → Skipping prompt templates (--create-users-only)")

    # 4. Admin Users
    for admin in config.admin_users:
        if admin.email:
            user_id = svc.ensure_user(admin.email, firm_id, is_admin=True)
            if user_id:
                svc.link_user_to_campaign(user_id, campaign_id)
                print(f"  ✓ Admin {admin.email} linked to campaign")

    # 5. Regular Users (from csv)
    users_csv = base_dir / "users.csv"
    if users_csv.exists():
        print(f"  → Processing users from {users_csv.name}...")
        try:
            with users_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    email = row.get("email", "").strip()
                    
                    if not email:
                        continue
                    
                    user_id = svc.ensure_user(email, firm_id, is_admin=False)
                    if user_id:
                        svc.link_user_to_campaign(user_id, campaign_id)
                        count += 1
                
                print(f"  ✓ Processed {count} users")
        except Exception as e:
            print(f"  ✗ Error processing users.csv: {e}")


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
    print("\n[1/5] Loading configuration...")
    try:
        config = load_unified_config(base_dir)
        print(f"  ✓ Campaign: {config.campaign.name} ({config.campaign.id})")
        print(f"  ✓ Language: {config.campaign.language}")
        print(f"  ✓ Action types: {config.get_action_types()}")
    except Exception as e:
        print(f"  ✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Load firm config (from parent directory or legacy)
    firm_config = load_firm_config(base_dir, config)
    if firm_config:
        print(f"  ✓ Firm: {firm_config.name}")
    else:
        print("  ⚠️ No firm configuration found")
        
    # Step 2: Supabase Onboarding
    print("\n[2/5] Supabase Sync...")
    if args.skip_supabase:
        print("  → Skipped (--skip-supabase)")
    else:
        try:
            run_supabase_onboarding(config, firm_config, base_dir, args.dry_run, args.local, args.create_users_only)
        except Exception as e:
            print(f"  ✗ Supabase sync failed: {e}")
            sys.exit(1)

    
    # Step 3: Generate campaign info (if enabled)
    print("\n[3/5] Campaign information generation...")
    if args.create_users_only:
        print("  → Skipped (--create-users-only)")
    elif args.skip_campaign_info:
        print("  → Skipped (--skip-campaign-info)")
    elif not config.prompts.generate_campaign_info:
        print("  → Disabled in config (prompts.generate_campaign_info = false)")
    else:
        generated_info_path = base_dir / "campaign_info" / f"{config.campaign.id}_generated_info.md"
        campaign_info = None

        if generated_info_path.exists():
            print(f"  → Found existing generated info: {generated_info_path.name}")
            if args.dry_run:
                print("  [DRY RUN] Would use existing campaign info")
            else:
                try:
                    with generated_info_path.open("r", encoding="utf-8") as f:
                        campaign_info = f.read()
                    print("  → Using existing file (skipping generation)")
                except Exception as e:
                    print(f"  ✗ Failed to read existing campaign info: {e}")
                    sys.exit(1)
        
        # If not found (or dry run didn't load it), try to generate (unless it existed and was dry run)
        if campaign_info is None and not generated_info_path.exists():
            campaign_info_files = get_campaign_info_files(base_dir)
            if not campaign_info_files:
                print("  → No files found in campaign_info/ directory, skipping")
            else:
                print(f"  → Found {len(campaign_info_files)} campaign info files: {[f.name for f in campaign_info_files]}")
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
                    except Exception as e:
                        print(f"  ✗ Failed to generate campaign info: {e}")
                        print("  ✗ Stopping - campaign info is required for prompts")
                        sys.exit(1)

        # Common steps: Preview and Upload
        if campaign_info:
            # Preview
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
            
            # Upload
            try:
                print(f"  → Uploading campaign info to Langfuse...")
                from services.prompt_generator import upload_campaign_info_prompt
                prompt_name = upload_campaign_info_prompt(config, campaign_info)
                print(f"  ✓ Uploaded as '{prompt_name}' to Langfuse")
            except Exception as e:
                print(f"  ✗ Failed to upload campaign info: {e}")
                sys.exit(1)
    
    # Step 4: Create prompts in Langfuse (if enabled)
    print("\n[4/5] Prompt creation in Langfuse...")
    if args.create_users_only:
        print("  → Skipped (--create-users-only)")
    elif args.skip_prompts:
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
    
    # Step 5: Generate datasets (if enabled and audio files exist)
    print("\n[5/5] Dataset generation from audio files...")
    if args.create_users_only:
        print("  → Skipped (--create-users-only)")
    elif args.skip_dataset:
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
