"""
Campaign Information Generator Service.

This service uses an LLM to generate structured campaign information
from raw campaign materials (product info, sales scripts, etc.).

The output follows a structured format with:
- Product facts
- Hook scripts
- Objection handling
- Closing techniques
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from services.llm_client import GeminiChatCompletionClient
from langfuse import Langfuse
from utils import load_env_value

if TYPE_CHECKING:
    from onboard import UnifiedConfig

# Deep research model for campaign info generation
DEEP_RESEARCH_MODEL = "gemini-3-pro-preview"


def _generate_with_deep_research(system_prompt: str, user_prompt: str) -> str:
    """Generate content using Gemini 3 Pro with full reasoning and web search.
    
    Uses google_search tool for web research and high thinking budget for full reasoning.
    """
    from google import genai
    from google.genai import types
    
    api_key = load_env_value("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    
    client = genai.Client(api_key=api_key)
    
    # Combine system and user prompts
    full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
    
    # Use google_search tool and high thinking for full reasoning
    response = client.models.generate_content(
        model=DEEP_RESEARCH_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=types.ThinkingConfig(thinking_budget=24576),  # Full reasoning
        ),
    )
    
    # Extract text from response
    text = getattr(response, "text", None)
    if text:
        return str(text)
    
    if getattr(response, "candidates", None):
        for candidate in response.candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for part in parts:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        return str(part_text)
    
    raise ValueError("Gemini returned no content")


# Langfuse prompt name for campaign information extraction
CAMPAIGN_INFO_EXTRACTOR_PROMPT = "campaign_information_extractor"

# Default prompt if Langfuse prompt is not available
DEFAULT_EXTRACTOR_PROMPT = """
Du er en ekspert i at analysere salgsmateriale og udtrække nøgleinformation.

Baseret på det vedlagte kampagnemateriale, generer en struktureret kampagnebeskrivelse med følgende sektioner:

## PRODUKTFAKTA
- Produktnavn
- Priser og abonnementer
- Bindingsperioder
- Særlige fordele

## HOOKS OG INDLEDNINGER
- Standard intro
- Attention hooks
- Behovsspørgsmål

## INDVENDINGSHÅNDTERING
- Almindelige indvendinger og svar
- Overbevisende argumenter

## CLOSING TEKNIKKER
- Overgang til data
- Antagelses-close
- Ordrebekræftelse

Brug det medfølgende sprog og stil fra materialet. Bevar de præcise tal og fakta.
"""


def _get_langfuse_client() -> Optional[Langfuse]:
    """Get or initialize the Langfuse client."""
    secret_key = load_env_value("LANGFUSE_SECRET_KEY")
    public_key = load_env_value("LANGFUSE_PUBLIC_KEY")
    host = load_env_value("LANGFUSE_HOST")

    if not secret_key or not public_key:
        return None

    return Langfuse(secret_key=secret_key, public_key=public_key, host=host)


def _get_extractor_prompt() -> str:
    """Get the campaign information extractor prompt from Langfuse or fallback to default."""
    try:
        langfuse = _get_langfuse_client()
        if langfuse:
            prompt = langfuse.get_prompt(CAMPAIGN_INFO_EXTRACTOR_PROMPT)
            return prompt.compile()
    except Exception as e:
        print(f"  → Could not fetch Langfuse prompt '{CAMPAIGN_INFO_EXTRACTOR_PROMPT}': {e}")
        print(f"  → Using default extractor prompt")
    
    return DEFAULT_EXTRACTOR_PROMPT


def _read_campaign_files(files: List[Path]) -> str:
    """Read and concatenate all campaign info files."""
    contents = []
    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()
            contents.append(f"### {file_path.name}\n\n{content}")
        except Exception as e:
            print(f"  → Warning: Could not read {file_path.name}: {e}")
    
    return "\n\n---\n\n".join(contents)


def generate_campaign_info(
    config: "UnifiedConfig",
    files: List[Path],
    output_dir: Path,
) -> str:
    """
    Generate campaign information from raw materials using LLM.
    
    Args:
        config: The unified campaign configuration
        files: List of paths to campaign info files
        output_dir: Directory to save the generated campaign info
        
    Returns:
        The generated campaign information as a string
    """
    print(f"  → Reading {len(files)} campaign info files...")
    raw_content = _read_campaign_files(files)
    
    if not raw_content.strip():
        raise ValueError("No content found in campaign info files")
    
    print(f"  → Fetching extractor prompt...")
    system_prompt = _get_extractor_prompt()
    
    user_prompt = f"""
Kampagne: {config.campaign.name}
Sprog: {config.campaign.language}

---

{raw_content}
"""
    
    print(f"  → Calling Gemini 3 Pro with full reasoning and web search...")
    response = _generate_with_deep_research(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    
    # Save the generated campaign info
    output_path = output_dir / f"{config.campaign.id}_generated_info.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(response)
    
    print(f"  → Saved generated campaign info to {output_path.name}")
    
    return response
