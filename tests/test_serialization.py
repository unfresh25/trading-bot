import json
import math

import pytest

from hybrid_trader.serialization import dump_json, json_safe


def test_json_safe_replaces_non_finite_with_none():
    payload = {"profit_factor": math.inf, "a": [1.0, -math.inf, float("nan")], "b": "ok"}
    safe = json_safe(payload)
    assert safe["profit_factor"] is None
    assert safe["a"] == [1.0, None, None]
    assert safe["b"] == "ok"


def test_dump_json_writes_strict_parseable_json_when_profit_factor_is_infinite(tmp_path):
    destination = tmp_path / "manifest.json"
    dump_json({"metrics": {"profit_factor": math.inf, "net_return": 0.02}}, destination)
    # A strict parser must accept the file: json.loads rejects the bare `Infinity` token.
    reloaded = json.loads(destination.read_text(encoding="utf-8"))
    assert reloaded["metrics"]["profit_factor"] is None
    assert reloaded["metrics"]["net_return"] == pytest.approx(0.02)


def test_dump_json_rejects_leftover_nan_loudly():
    # allow_nan=False guarantees no silent Infinity/NaN token can ever be written.
    with pytest.raises(ValueError):
        json.dumps({"x": math.nan}, allow_nan=False)
