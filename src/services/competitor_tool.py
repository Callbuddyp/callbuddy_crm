"""Competitor tool service for on-demand competitor lookups during state patching."""

import time
from typing import Any, Callable, Dict, List, Optional

from services.supabase_client import SupabaseService


def _info(msg: str) -> None:
    print(f"\033[36m[info]\033[0m {msg}")


def _warn(msg: str) -> None:
    print(f"\033[33m[warn]\033[0m {msg}")


class CompetitorToolService:
    """Manages competitor tool definitions and handles tool calls.

    Fetches competitor IDs at init (lightweight), builds tool definitions,
    and handles on-demand lookups with state injection.
    """

    def __init__(self, supabase_service: SupabaseService, campaign_id: str) -> None:
        self._supabase = supabase_service
        self._campaign_id = campaign_id
        self._competitors: List[Dict[str, str]] = []

        t0 = time.perf_counter()
        self._competitors = supabase_service.get_campaign_competitor_ids(campaign_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _info(f"Fetched {len(self._competitors)} competitor IDs in {elapsed_ms:.0f}ms")

    @property
    def has_competitors(self) -> bool:
        return len(self._competitors) > 0

    def should_offer_tool(self, state: Dict[str, Any]) -> bool:
        """True if there are competitors not yet fetched into state."""
        if not self._competitors:
            return False
        backgrounds = state.get("competitor_backgrounds", {})
        return any(
            c["competitor_id"] not in backgrounds
            for c in self._competitors
        )

    def _get_name(self, competitor_id: str) -> str:
        """Get display name for a competitor_id."""
        for c in self._competitors:
            if c["competitor_id"] == competitor_id:
                return c.get("name", competitor_id)
        return competitor_id

    def build_tool_definition(self, provider: str) -> Any:
        """Build a tool declaration for lookup_competitor.

        Args:
            provider: "fireworks" returns OpenAI-style dict,
                      "gemini"/"vertex" returns a genai FunctionDeclaration.
        """
        enum_values = [c["competitor_id"] for c in self._competitors]

        if provider in ("fireworks", "fireworks-lite"):
            return {
                "type": "function",
                "function": {
                    "name": "lookup_competitor",
                    "description": (
                        "Look up detailed competitive intelligence for a specific competitor. "
                        "Call this when the customer mentions a competitor by name."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "competitor_id": {
                                "type": "string",
                                "enum": enum_values,
                                "description": "The competitor to look up",
                            },
                        },
                        "required": ["competitor_id"],
                    },
                },
            }

        from google.genai import types

        return types.FunctionDeclaration(
            name="lookup_competitor",
            description=(
                "Look up detailed competitive intelligence for a specific competitor. "
                "Call this when the customer mentions a competitor by name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "competitor_id": {
                        "type": "string",
                        "enum": enum_values,
                        "description": "The competitor to look up",
                    },
                },
                "required": ["competitor_id"],
            },
        )

    def handle_tool_call(self, competitor_id: str, state: Dict[str, Any]) -> str:
        """Fetch competitor description from Supabase and inject into state.

        Returns a brief confirmation string (sent back to the LLM).
        """
        t0 = time.perf_counter()
        description = self._supabase.get_competitor_description(
            self._campaign_id, competitor_id
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if description is None:
            _warn(f"No data found for competitor '{competitor_id}' ({elapsed_ms:.0f}ms)")
            return f"No data found for competitor '{competitor_id}'."

        # Inject into state
        if "competitor_backgrounds" not in state:
            state["competitor_backgrounds"] = {}
        state["competitor_backgrounds"][competitor_id] = description

        name = self._get_name(competitor_id)
        _info(f"Fetched competitor '{competitor_id}' in {elapsed_ms:.0f}ms ({len(description)} chars)")
        return f"Added competitor background for {name}"
