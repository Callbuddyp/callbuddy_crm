from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from services.llm import call_scenario_selector_for_type
from models.scenario_test_case import ScenarioTestCase


@dataclass
class Utterance:
    speaker_id: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def cleaned_text(self) -> str:
        return self.text.strip()


class Conversation:
    def __init__(self, utterances: List[Utterance], language: str = "Dansk"):
        self.language = language
        self.utterances = sorted(utterances, key=lambda utt: (utt.start_ms, utt.end_ms))

    @staticmethod
    def _speaker_id_from_token(token: dict, fallback: Optional[int]) -> int:
        speaker_value = token.get("speaker", fallback)
        try:
            return int(speaker_value) if speaker_value is not None else -1
        except (TypeError, ValueError):
            return -1

    @staticmethod
    def _time_from_token(token: dict, key: str) -> Optional[int]:
        for candidate in (f"{key}_time_ms", f"{key}_ms", key):
            value = token.get(candidate)
            if isinstance(value, (int, float)):
                return int(value)
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()

    @classmethod
    def from_soniox_transcription(cls, transcription: dict, language: str = "Dansk") -> "Conversation":
        tokens = transcription.get("tokens") or transcription.get("final_tokens") or transcription.get("finalTokens")
        if tokens is None and "transcript" in transcription:
            transcript_block = transcription["transcript"]
            tokens = (
                transcript_block.get("tokens")
                or transcript_block.get("final_tokens")
                or transcript_block.get("finalTokens")
            )
        if tokens is None:
            raise ValueError("Transcription response does not contain tokens.")
        return cls.from_soniox_tokens(tokens, language=language)

    @classmethod
    def from_soniox_tokens(cls, tokens: Iterable[dict], language: str = "Dansk") -> "Conversation":
        utterances: List[Utterance] = []
        current_speaker: Optional[int] = None
        current_parts: List[str] = []
        start_ms: Optional[int] = None
        end_ms: Optional[int] = None

        def flush() -> None:
            if current_speaker is None:
                return
            normalized_text = cls._normalize_text("".join(current_parts))
            if not normalized_text:
                return
            utterances.append(
                Utterance(
                    speaker_id=current_speaker,
                    start_ms=start_ms or 0,
                    end_ms=end_ms or start_ms or 0,
                    text=normalized_text,
                )
            )

        for token in tokens:
            if token.get("translation_status") == "translation":
                continue
            text = str(token.get("text", "") or "")
            if text == "":
                continue

            speaker = cls._speaker_id_from_token(token, current_speaker)
            token_start = cls._time_from_token(token, "start")
            token_end = cls._time_from_token(token, "end") or cls._time_from_token(token, "stop")

            if current_speaker is None:
                current_speaker = speaker
                current_parts = [text]
                start_ms = token_start
                end_ms = token_end or token_start
                continue

            if speaker == current_speaker:
                current_parts.append(text)
                if token_start is not None and (start_ms is None or token_start < start_ms):
                    start_ms = token_start
                if token_end is not None:
                    end_ms = max(end_ms or token_end, token_end)
            else:
                flush()
                current_speaker = speaker
                current_parts = [text]
                start_ms = token_start
                end_ms = token_end or token_start

        flush()
        return cls(utterances=utterances, language=language)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "utterances": [
                {
                    "speaker_id": utt.speaker_id,
                    "start_ms": utt.start_ms,
                    "end_ms": utt.end_ms,
                    "text": utt.text,
                }
                for utt in self.utterances
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        utterances_data = data.get("utterances") or []
        utterances = [
            Utterance(
                speaker_id=int(item["speaker_id"]),
                start_ms=int(item.get("start_ms", 0)),
                end_ms=int(item.get("end_ms", item.get("start_ms", 0))),
                text=str(item.get("text", "")),
            )
            for item in utterances_data
        ]
        language = data.get("language") or "Dansk"
        return cls(utterances=utterances, language=language)

    def get_utterance(self, index: int) -> Utterance:
        if index < 0 or index >= len(self.utterances):
            raise ValueError(f"anchor_index {index} is outside the utterance range")
        return self.utterances[index]

    def generate_test_cases(
        self,
        is_speaker_one_seller: bool,
        audio_name: Optional[str] = None,
        audio_stem: Optional[str] = None,
        generate_ai_cases: bool = False,
        soniox_metadata: Optional[dict] = None,
        model: Optional[str] = None,
        action_types: Optional[List[str]] = None,
        vad_suggestion_indices: Optional[List[int]] = None,
    ) -> List[ScenarioTestCase]:
        """Generate test cases for the conversation.
        
        Args:
            is_speaker_one_seller: Whether speaker 1 is the seller
            audio_name: Original audio filename
            audio_stem: Audio filename without extension
            generate_ai_cases: Whether to generate AI suggestion cases for every seller turn (legacy)
            soniox_metadata: Optional Soniox transcription metadata
            model: Optional LLM model override
            action_types: List of action types to generate. If None, uses legacy behavior.
            vad_suggestion_indices: List of utterance indices for VAD-detected AI suggestions
        """
        payload = {
            "audio_name": audio_name or "",
            "audio_stem": audio_stem or "",
            "conversation": self.to_dict(),
        }
        if soniox_metadata:
            payload["soniox"] = soniox_metadata

        cases: List[ScenarioTestCase] = []
        
        if action_types:
            # New behavior: call type-specific selectors for each action type
            for action_type in action_types:
                if action_type == "ai_suggestion":
                    # ai_suggestion is handled via VAD, skip LLM call
                    continue
                try:
                    type_cases = call_scenario_selector_for_type(
                        conversation_payload=payload,
                        action_type=action_type,
                        model=model,
                    )
                    cases.extend(type_cases)
                    print(f"  -> Found {len(type_cases)} '{action_type}' scenarios")
                except Exception as exc:
                    print(f"  -> Error detecting '{action_type}': {exc}")
        
        # Generate AI suggestion cases from VAD detection
        if vad_suggestion_indices:
            seller_id = 1 if is_speaker_one_seller else 2
            customer_id = 2 if is_speaker_one_seller else 1
            
            for idx in vad_suggestion_indices:
                if idx < 0 or idx >= len(self.utterances):
                    continue
                
                # Find the customer text at this index
                customer_utt = self.utterances[idx]
                
                # Find the next seller utterance after this point
                seller_action = ""
                for j in range(idx + 1, len(self.utterances)):
                    if self.utterances[j].speaker_id == seller_id:
                        seller_action = self.utterances[j].text
                        break
                
                if not seller_action:
                    continue
                
                test_case = ScenarioTestCase(
                    anchor_utterance_index=idx,
                    suggested_tests=["ai_suggestion"],
                    customer_text=customer_utt.text,
                    seller_action=seller_action,
                )
                cases.append(test_case)
            
            print(f"  -> Generated {len(vad_suggestion_indices)} 'ai_suggestion' cases from VAD")
        
        # Legacy: Generate AI suggestion cases for every seller turn
        elif generate_ai_cases:
            seller_id = 1 if is_speaker_one_seller else 2
            for idx, utterance in enumerate(self.utterances):
                if idx < 1:
                    continue
                if utterance.speaker_id == seller_id:
                    test_case = ScenarioTestCase(
                        anchor_utterance_index=idx - 1,
                        suggested_tests=["ai_suggestion"],
                        customer_text=self.utterances[idx - 1].text,
                        seller_action=utterance.text,
                    )
                    cases.append(test_case)

        return cases

 