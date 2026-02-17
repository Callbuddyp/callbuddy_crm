"""Voice Activity Detection service using Silero VAD v5.

Detects speech/silence segments in audio files to programmatically
identify AI suggestion points.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Default configuration constants
DEFAULT_SPEECH_MINIMUM_DURATION_MS = 2000  # 2 seconds
DEFAULT_SILENCE_BEFORE_AI_MS = 500  # 500ms


@dataclass
class VADConfig:
    """Configuration for VAD-based AI suggestion detection."""
    speech_minimum_duration_ms: int = DEFAULT_SPEECH_MINIMUM_DURATION_MS
    silence_before_ai_ms: int = DEFAULT_SILENCE_BEFORE_AI_MS
    
    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "VADConfig":
        """Create VADConfig from a dictionary (e.g., from customer config)."""
        if not data:
            return cls()
        return cls(
            speech_minimum_duration_ms=data.get("speech_minimum_duration_ms", DEFAULT_SPEECH_MINIMUM_DURATION_MS),
            silence_before_ai_ms=data.get("silence_before_ai_ms", DEFAULT_SILENCE_BEFORE_AI_MS),
        )


@dataclass
class SpeechSegment:
    """A segment of detected speech."""
    start_ms: int
    end_ms: int
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


# Lazy-loaded model cache
_vad_model = None
_vad_utils = None


def load_vad_model():
    """Load and cache Silero VAD v5 model."""
    global _vad_model, _vad_utils
    
    if _vad_model is not None:
        return _vad_model, _vad_utils
    
    try:
        import torch
        torch.set_num_threads(1)
        
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
        return model, utils
    except Exception as exc:
        print(f"[VAD] Failed to load Silero VAD model: {exc}")
        raise


def get_speech_timestamps(audio_path: Path, sample_rate: int = 16000) -> List[SpeechSegment]:
    """Get speech timestamps from an audio file using Silero VAD.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (Silero VAD works best at 16kHz)
        
    Returns:
        List of SpeechSegment with start/end times in milliseconds
    """
    model, utils = load_vad_model()
    get_speech_ts, _, read_audio, _, _ = utils
    
    # Read and resample audio
    wav = read_audio(str(audio_path), sampling_rate=sample_rate)
    
    # Get speech timestamps (in samples)
    speech_timestamps = get_speech_ts(
        wav,
        model,
        sampling_rate=sample_rate,
        return_seconds=False,  # Return in samples for precision
    )
    
    # Convert samples to milliseconds
    segments = []
    for ts in speech_timestamps:
        start_ms = int(ts['start'] / sample_rate * 1000)
        end_ms = int(ts['end'] / sample_rate * 1000)
        segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))
    
    return segments


def find_ai_suggestion_points(
    speech_segments: List[SpeechSegment],
    config: VADConfig,
) -> List[int]:
    """Find timestamps (in ms) where AI suggestions should be triggered.
    
    AI suggestion points are identified when:
    1. A speech segment lasted at least `speech_minimum_duration_ms`
    2. There is a silence gap of at least `silence_before_ai_ms` after it
    
    Args:
        speech_segments: List of speech segments from VAD
        config: VAD configuration with thresholds
        
    Returns:
        List of timestamps (in ms) where AI suggestions should be shown
    """
    suggestion_points = []
    
    for i, segment in enumerate(speech_segments):
        # Check if speech segment is long enough
        if segment.duration_ms < config.speech_minimum_duration_ms:
            continue
        
        # Calculate silence duration after this segment
        if i + 1 < len(speech_segments):
            next_segment = speech_segments[i + 1]
            silence_duration = next_segment.start_ms - segment.end_ms
        else:
            # Last segment - assume sufficient silence after
            silence_duration = config.silence_before_ai_ms
        
        # Check if silence is long enough
        if silence_duration >= config.silence_before_ai_ms:
            # The suggestion point is at the end of the speech segment
            suggestion_points.append(segment.end_ms)
    
    return suggestion_points


def map_timestamps_to_utterances(
    suggestion_points_ms: List[int],
    utterances: List[dict],
    tolerance_ms: int = 500,
) -> List[int]:
    """Map AI suggestion timestamps to utterance indices.
    
    Finds the utterance that ends closest to each suggestion point.
    
    Args:
        suggestion_points_ms: List of timestamps where AI suggestions should trigger
        utterances: List of utterance dicts with start_ms and end_ms
        tolerance_ms: Maximum allowed difference between suggestion point and utterance end
        
    Returns:
        List of utterance indices where AI suggestions should be generated
    """
    utterance_indices = []
    
    for point_ms in suggestion_points_ms:
        best_idx = None
        best_diff = float('inf')
        
        for idx, utt in enumerate(utterances):
            end_ms = utt.get('end_ms', 0)
            diff = abs(end_ms - point_ms)
            
            if diff < best_diff and diff <= tolerance_ms:
                best_diff = diff
                best_idx = idx
        
        if best_idx is not None and best_idx not in utterance_indices:
            utterance_indices.append(best_idx)
    
    return sorted(utterance_indices)


def detect_ai_suggestion_utterances(
    audio_path: Path,
    utterances: List[dict],
    config: Optional[VADConfig] = None,
) -> List[int]:
    """High-level function to detect AI suggestion points for an audio file.
    
    Args:
        audio_path: Path to the audio file
        utterances: List of utterance dicts from transcription
        config: Optional VAD configuration
        
    Returns:
        List of utterance indices where AI suggestions should be generated
    """
    if config is None:
        config = VADConfig()
    
    print(f"  -> Running VAD on {audio_path.name}...")
    
    try:
        # Get speech segments from VAD
        speech_segments = get_speech_timestamps(audio_path)
        print(f"  -> Found {len(speech_segments)} speech segments")
        
        # Find suggestion points based on speech/silence pattern
        suggestion_points = find_ai_suggestion_points(speech_segments, config)
        print(f"  -> Identified {len(suggestion_points)} potential AI suggestion points")
        
        # Map to utterance indices
        utterance_indices = map_timestamps_to_utterances(suggestion_points, utterances)
        print(f"  -> Mapped to {len(utterance_indices)} utterance indices")
        
        return utterance_indices
        
    except Exception as exc:
        print(f"  -> VAD error: {exc}")
        return []
