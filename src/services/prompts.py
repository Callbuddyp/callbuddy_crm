from __future__ import annotations

from typing import Dict, Optional
from langfuse import Langfuse
from utils import load_env_value

# Mapping from action type to Langfuse prompt name
ACTION_TYPE_TO_PROMPT_NAME: Dict[str, str] = {
    "objection": "gen_objection_dataset",
    "close": "gen_close_dataset",
    "opening": "gen_opening_dateset",
    "price_comparison": "gen_price_comparison_dataset",
}

ACTION_TYPES = list(ACTION_TYPE_TO_PROMPT_NAME.keys())

# Langfuse client singleton
_langfuse_client: Optional[Langfuse] = None


def _get_langfuse_client() -> Langfuse:
    """Get or initialize the Langfuse client for prompt fetching."""
    global _langfuse_client
    if _langfuse_client is None:
        secret_key = load_env_value("LANGFUSE_SECRET_KEY")
        public_key = load_env_value("LANGFUSE_PUBLIC_KEY")
        host = load_env_value("LANGFUSE_HOST")

        if not secret_key or not public_key:
            raise ValueError(
                "LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY missing. "
                "Cannot fetch prompts from Langfuse."
            )

        _langfuse_client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host,
        )
    return _langfuse_client


def get_prompt_for_action_type(action_type: str) -> str:
    """Fetch the latest production prompt from Langfuse for the given action type.
    
    Args:
        action_type: The action type (e.g., 'objection', 'close', 'opening')
        
    Returns:
        The compiled prompt text from Langfuse
        
    Raises:
        ValueError: If action_type is not recognized
    """
    if action_type not in ACTION_TYPE_TO_PROMPT_NAME:
        raise ValueError(
            f"Unknown action type: {action_type}. "
            f"Valid types: {list(ACTION_TYPE_TO_PROMPT_NAME.keys())}"
        )

    prompt_name = ACTION_TYPE_TO_PROMPT_NAME[action_type]
    langfuse = _get_langfuse_client()

    # Fetch the production version of the prompt
    prompt = langfuse.get_prompt(prompt_name)
    
    # Compile the prompt (in case it has variables, though we don't use any here)
    compiled_prompt = prompt.compile()
    
    print(f"  -> Fetched prompt '{prompt_name}' (version: {prompt.version})")
    
    return compiled_prompt


SCENARIO_SELECTOR_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "scenario_selector_response",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "anchor_utterance_index": {"type": "integer", "minimum": 0},
                    "suggested_tests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "customer_text": {"type": "string"},
                    "seller_action": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "description": "A brief explanation of why this scenario was identified and description of the problem",
                    },
                },
                "required": ["anchor_utterance_index", "suggested_tests", "customer_text", "seller_action", "reason"],
                "additionalProperties": False,
            },
        },
        "strict": False,
    },
}
