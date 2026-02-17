from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

from utils import load_env_value
from langfuse import Langfuse
import time


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate limit responses coming from Langfuse (status code 429)."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    message = str(exc).lower()
    return "429" in message and ("rate limit" in message or "too many requests" in message)


def _with_rate_limit_backoff(
    operation: Callable[[], None],
    *,
    max_attempts: int = 7,
    initial_delay: float = 1.0,
) -> None:
    """Retry an operation with exponential backoff when hitting rate limits."""
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if _is_rate_limit_error(exc) and attempt < max_attempts:
                print(f"[langfuse] Rate limited (attempt {attempt}/{max_attempts}); retrying in {delay} seconds.")
                time.sleep(delay)
                delay *= 2
                continue
            raise


def init_langfuse(push_to_langfuse: bool) -> Optional[Langfuse]:
    if not push_to_langfuse:
        print("langfuse disabled")
        return None
    if Langfuse is None:
        print("Langfuse SDK is not installed; skipping dataset upload.")
        return None

    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    host = os.environ.get("LANGFUSE_HOST")
    secret_key = load_env_value("LANGFUSE_SECRET_KEY") 
    public_key = load_env_value("LANGFUSE_PUBLIC_KEY")
    host = load_env_value("LANGFUSE_HOST")


    if not secret_key or not public_key:
        print("LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY missing; skipping dataset upload.")
        return None

    return Langfuse(secret_key=secret_key, public_key=public_key, host=host)


def ensure_dataset(types, langfuse_client: Optional[Langfuse], dataset_name: str, language: str) -> None:
    if langfuse_client is None:
        return
    try:
        
        time.sleep(1)
        for prompttype in types:
            dataset_name_type = dataset_name + "_" + prompttype
            print(dataset_name_type)
            _with_rate_limit_backoff(
                lambda: langfuse_client.create_dataset(
                    name=dataset_name_type,
                    description="Auto-generated from Soniox transcriptions.",
                    metadata={"language": language},
                )
            )
    except Exception as exc:
        # Dataset might already exist; log and continue.
        print(f"[langfuse] create_dataset skipped: {exc}")


def push_dataset_items(
    langfuse_client: Optional[Langfuse],
    dataset_name: str,
    items: List[dict],
    conversation_name: str,
) -> int:
    if langfuse_client is None or not items:
        return 0

    created = 0
    for idx, item in enumerate(items):
        metadata = dict(item.get("metadata", {}))
        input = dict(item.get("input", {}))
        tests = input['suggested_tests']
        metadata["conversation"] = conversation_name
        time.sleep(1)
        try:
            for test in tests: 
                dataset_name_type = dataset_name + "_" + test
                print(f"  -> Pushing to dataset: {dataset_name_type}")
                _with_rate_limit_backoff(
                    lambda: langfuse_client.create_dataset_item(
                        dataset_name=dataset_name_type,
                        input=item["input"],
                        metadata=metadata,
                        expected_output=item['expected_output'],
                    )
                )
                created += 1
        except Exception as exc:
            print(f"[langfuse] Failed to push item for {conversation_name}: {exc}")
    return created
