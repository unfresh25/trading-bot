import pandas as pd

from hybrid_trader.diagnostics import prediction_diagnostics


def test_prediction_diagnostics_cover_classes_calibration_and_groups():
    index = pd.DatetimeIndex(
        ["2024-12-31T00:00:00Z", "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"]
    )
    predictions = pd.DataFrame(
        {
            "label": [-1, 0, 1, 1],
            "probability_down": [0.8, 0.2, 0.1, 0.2],
            "probability_neutral": [0.1, 0.6, 0.2, 0.1],
            "probability_up": [0.1, 0.2, 0.7, 0.7],
            "fold": [0, 0, 1, 1],
            "regime": ["trend", "sideways", "trend", "trend"],
        },
        index=index,
    )

    result = prediction_diagnostics(predictions)

    assert result["class_distribution"] == {"DOWN": 0.25, "NEUTRAL": 0.25, "UP": 0.5}
    assert result["probability"]["brier_up"] >= 0
    assert result["probability"]["log_loss"] >= 0
    assert result["probability"]["baseline_brier_up"] >= 0
    assert result["probability"]["baseline_log_loss"] >= 0
    assert "brier_skill_score" in result["probability"]
    assert sum(item["count"] for item in result["calibration_up"]) == 4
    assert set(result["by_fold"]) == {"0", "1"}
    assert set(result["by_year"]) == {"2024", "2025"}
    assert set(result["by_regime"]) == {"trend", "sideways"}
