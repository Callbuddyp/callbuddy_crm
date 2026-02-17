from pathlib import Path
from typing import Optional


def load_env_value(key: str, env_path: Optional[str | Path] = None) -> Optional[str]:
    """Lightweight .env loader (quotes stripped) to avoid extra deps."""
    candidate_paths = [
        Path(env_path) if env_path else None,
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for path in candidate_paths:
        if path is None or not path.exists():
            continue
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, _, value = line.partition("=")
                if name.strip() != key:
                    continue
                return value.strip().strip('"').strip("'")
        except Exception:
            continue
    return None
