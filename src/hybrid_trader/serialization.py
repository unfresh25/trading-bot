from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with None so the result is RFC 8259 JSON.

    ``json.dumps`` emits the bare tokens ``Infinity``/``NaN`` for non-finite floats,
    which strict parsers reject and ``jq`` silently coerces to ``1.79e308``. Metrics
    such as ``profit_factor`` are legitimately ``inf`` when there are no losing trades,
    so the value is preserved in memory and only sanitised at the serialisation edge.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def dump_json(payload: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Serialise ``payload`` to ``path`` as strict JSON, failing loudly on any leftover NaN/inf."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(payload), indent=indent, allow_nan=False, default=str),
        encoding="utf-8",
    )
    return destination
