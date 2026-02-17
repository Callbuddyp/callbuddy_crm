from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from services.llm_client import scenario_selector_gemini_llm
from services.prompts import get_prompt_for_action_type
from models.scenario_test_case import ScenarioTestCase


def _with_utterance_indices(payload: Dict[str, Any]) -> Dict[str, Any]:
    conversation = payload.get("conversation") or {}
    utterances = conversation.get("utterances") or []
    conversation_payload = {
        "language": conversation.get("language") or "",
        "utterances": [],
    }
    for idx, utterance in enumerate(utterances):
        if not isinstance(utterance, dict):
            continue
        entry = dict(utterance)
        entry["utterance_index"] = idx
        conversation_payload["utterances"].append(entry)
    result = {
        "audio_name": payload.get("audio_name") or "",
        "audio_stem": payload.get("audio_stem") or "",
        "conversation": conversation_payload,
    }
    if "soniox" in payload:
        result["soniox"] = payload["soniox"]
    return result


def _strip_json_code_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_array(text: str) -> Optional[str]:
    match = re.search(r"\[[\s\S]*\]", text)
    return match.group(0) if match else None


def _coerce_entries(parsed: Any) -> List[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("scenarios", "cases", "items", "data", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("Scenario selector response must be a JSON array.")


def parse_scenario_selector_response(raw_text: str) -> List[ScenarioTestCase]:

    print(raw_text)
    cleaned = _strip_json_code_fence(raw_text)
    parsed: Any = None
    last_error: Optional[Exception] = None
    for candidate in filter(None, (cleaned, _extract_json_array(cleaned))):
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
    if parsed is None:
        raise ValueError(f"Could not parse scenario selector response: {last_error}")
    entries = _coerce_entries(parsed)
    return [ScenarioTestCase.parse_obj(entry) for entry in entries]



def call_scenario_selector_for_type(
    conversation_payload: Dict[str, Any],
    action_type: str,
    model: Optional[str] = None,
) -> List[ScenarioTestCase]:
    """Call the scenario selector for a specific action type.
    
    Fetches the type-specific prompt from Langfuse to identify
    only scenarios of that particular type.
    
    Args:
        conversation_payload: The conversation data to analyze
        action_type: The action type to detect (e.g., 'objection', 'close')
        model: Optional model override
        
    Returns:
        List of ScenarioTestCase objects for the specified action type
    """
    # Validation now happens inside get_prompt_for_action_type
    prompt = get_prompt_for_action_type(action_type)
    payload = _with_utterance_indices(conversation_payload)
    
    print(f"  -> Detecting '{action_type}' scenarios...")
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    content = scenario_selector_gemini_llm.generate(
        messages=messages,
        model=model,
    )
    return parse_scenario_selector_response(str(content))
