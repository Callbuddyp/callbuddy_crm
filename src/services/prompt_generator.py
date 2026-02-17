"""
Prompt Generator Service.

This service creates campaign-specific prompts in Langfuse.
Supports two prompt structures:

1. Standard: Combines multiple prompt components
   - Base system prompt (salescall_system1)
   - Campaign-specific information ({campaign_id}_info)
   - Output format prompt (output_markdown_suggestions)
   - Action-specific prompt (objection_prompt, close_prompt, etc.)

2. Custom: Uses inline prompt_content with optional features
   - Can include web search, custom formatting, etc.

Each prompt also includes a config block for frontend settings:
   - use_reasoning: Enable reasoning/thinking
   - use_internet: Enable web search
   - output_type: Response format ("text", "markdown", etc.)
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from langfuse import Langfuse
from utils import load_env_value

if TYPE_CHECKING:
    from scripts.onboard import UnifiedConfig, ActionPromptConfig


def _get_langfuse_client() -> Langfuse:
    """Get or initialize the Langfuse client."""
    secret_key = load_env_value("LANGFUSE_SECRET_KEY")
    public_key = load_env_value("LANGFUSE_PUBLIC_KEY")
    host = load_env_value("LANGFUSE_HOST")

    if not secret_key or not public_key:
        raise ValueError(
            "LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY missing. "
            "Cannot create prompts in Langfuse."
        )

    return Langfuse(secret_key=secret_key, public_key=public_key, host=host)


def _build_prompt_reference(prompt_name: str, label: str = "production") -> str:
    """Build a Langfuse prompt reference tag for composability.
    
    Uses Langfuse's native composability syntax that gets resolved at compile-time.
    See: https://langfuse.com/docs/prompt-management/features/composability
    
    Args:
        prompt_name: Name of the prompt to reference
        label: Label to use for resolution (default: "production")
        
    Returns:
        A Langfuse prompt reference tag
    """
    return f"@@@langfusePrompt:name={prompt_name}|label={label}@@@"


def _get_existing_prompt_type(langfuse: Langfuse, name: str) -> Optional[str]:
    """Check if a prompt exists and return its type, or None if it doesn't exist."""
    try:
        prompt = langfuse.get_prompt(name)
        # Langfuse returns TextPromptClient or ChatPromptClient based on prompt type
        class_name = type(prompt).__name__
        if 'Text' in class_name:
            return 'text'
        elif 'Chat' in class_name:
            return 'chat'
        return None
    except Exception:
        # Prompt doesn't exist
        return None


def _convert_chat_to_text(messages: List[Dict[str, str]]) -> str:
    """Convert chat-format messages to a single text string."""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def _create_or_update_prompt(
    langfuse: Langfuse,
    name: str,
    prompt_content: Union[str, List[Dict[str, str]]],
    config: Dict[str, Any] = None,
    labels: List[str] = None,
    prompt_type: str = "text",
) -> None:
    """Create or update a prompt in Langfuse with config.
    
    Args:
        langfuse: Langfuse client
        name: Prompt name
        prompt_content: String for text prompts, list of message dicts for chat prompts
        config: Frontend configuration (use_reasoning, use_internet, output_type, etc.)
        labels: Langfuse labels (default: ["production"])
        prompt_type: "text" or "chat"
    """
    if labels is None:
        labels = ["production"]
    
    # Build the config object
    prompt_config = config or {}
    
    # Check if prompt already exists to handle type conversion
    existing_type = _get_existing_prompt_type(langfuse, name)
    
    if existing_type is not None and existing_type != prompt_type:
        print(f"    ⚠ Prompt '{name}' exists with type '{existing_type}', requested '{prompt_type}'")
        # Convert content format if needed
        if existing_type == "text" and prompt_type == "chat":
            # Convert chat messages to text format
            prompt_content = _convert_chat_to_text(prompt_content)
            print(f"    → Converting chat format to text for compatibility")
        elif existing_type == "chat" and prompt_type == "text":
            # Convert text to chat format (single system message)
            prompt_content = [{"role": "system", "content": prompt_content}]
            print(f"    → Converting text format to chat for compatibility")
        prompt_type = existing_type
    
    try:
        # Create a new prompt version
        langfuse.create_prompt(
            name=name,
            prompt=prompt_content,
            config=prompt_config,
            labels=labels,
            type=prompt_type,
            commit_message="created by onboarding script"
        )
        print(f"    ✓ Created/updated prompt: {name} (type: {prompt_type})")
        if prompt_config:
            print(f"      Config: {prompt_config}")
    except Exception as e:
        print(f"    ✗ Failed to create prompt '{name}': {e}")
        raise


def _build_standard_prompt(
    action_config: "ActionPromptConfig",
    campaign_id: str,
) -> List[Dict[str, str]]:
    """Build a standard chat prompt using Langfuse composability references.
    
    Instead of fetching and embedding prompt content, this function generates
    Langfuse prompt reference tags that get resolved at compile-time.
    
    See: https://langfuse.com/docs/prompt-management/features/composability
    
    Returns a list of message dicts for chat format with reference tags.
    """
    system_parts = []
    user_parts = []
    
    # Process system components - create reference tags
    for component in action_config.components.get("system", []):
        # Replace {campaign_id} placeholder if present
        resolved_name = component.replace("{campaign_id}", campaign_id)
        system_parts.append(_build_prompt_reference(resolved_name))
    
    # Process user components - create reference tags
    for component in action_config.components.get("user", []):
        resolved_name = component.replace("{campaign_id}", campaign_id)
        user_parts.append(_build_prompt_reference(resolved_name))
    
    # Build chat messages with reference tags
    messages = []
    
    if system_parts:
        messages.append({
            "role": "system",
            "content": "\n\n".join(system_parts)
        })
    
    if user_parts:
        messages.append({
            "role": "user",
            "content": "\n\n".join(user_parts)
        })
    
    return messages


def _build_custom_prompt(action_config: "ActionPromptConfig") -> str:
    """Build a custom prompt from inline content.
    
    Custom prompts use the prompt_content field directly.
    """
    content = action_config.prompt_content or ""
    
    # For custom prompts, we typically use a simpler structure
    # The prompt_content usually contains the full prompt with {{transcript}} placeholder
    return content


def create_campaign_prompts(config: "UnifiedConfig") -> Dict[str, str]:
    """
    Create campaign-specific prompts in Langfuse.
    
    Supports both standard (component-based) and custom (inline) prompt structures.
    Each prompt includes frontend config (use_reasoning, use_internet, output_type).
    
    Args:
        config: The unified campaign configuration
        
    Returns:
        Dictionary mapping action types to created prompt names
    """
    langfuse = _get_langfuse_client()
    created_prompts = {}
    campaign_id = config.campaign.id
    
    print(f"  → Creating action prompts...")
    
    for action_type, action_config in config.prompts.actions.items():
        if not action_config.enabled:
            print(f"    → Skipping {action_type} (disabled)")
            continue
        
        print(f"    → Processing {action_type}...")
        
        # Build prompt based on structure type
        if action_config.structure == "custom":
            prompt_content = _build_custom_prompt(action_config)
            prompt_type = "text"
        else:
            # Standard structure - returns chat messages with composability references
            prompt_content = _build_standard_prompt(action_config, campaign_id)
            prompt_type = "chat"
        
        # Get the frontend config for this action
        frontend_config = dict(action_config.frontend_config) if action_config.frontend_config else {}
        
        try:
            _create_or_update_prompt(
                langfuse=langfuse,
                name=action_config.langfuse_name,
                prompt_content=prompt_content,
                config=frontend_config,
                prompt_type=prompt_type,
            )
            created_prompts[action_type] = action_config.langfuse_name
        except Exception as e:
            print(f"    ✗ Failed to create {action_type} prompt: {e}")
            raise
    
    return created_prompts


def upload_campaign_info_prompt(
    config: "UnifiedConfig",
    campaign_info_content: str,
) -> str:
    """
    Upload the generated campaign info as a prompt to Langfuse.
    
    Args:
        config: The unified campaign configuration
        campaign_info_content: The generated campaign info markdown
        
    Returns:
        The name of the created prompt
    """
    langfuse = _get_langfuse_client()
    prompt_name = f"{config.campaign.id}_info"
    
    _create_or_update_prompt(
        langfuse=langfuse,
        name=prompt_name,
        prompt_content=campaign_info_content,
        config={"type": "campaign_info"},
    )
    
    return prompt_name
