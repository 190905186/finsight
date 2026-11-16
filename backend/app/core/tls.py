from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_ca_cert(value: str | None) -> str | bool:
    """Return an httpx-compatible verify argument.

    Accepts an absolute/relative path to a PEM file. Relative paths are resolved
    against the repository root. Empty values fall back to system trust.
    """
    if value is None:
        return True
    raw = value.strip()
    if not raw:
        return True

    # Inline PEM bodies are not supported as httpx verify paths.
    if "BEGIN CERTIFICATE" in raw:
        raise ValueError(
            "CA cert must be a file path (e.g. llm/vm04-ca.crt), not inline PEM text"
        )

    path = Path(raw)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CA certificate not found: {path}")
    return str(path)
