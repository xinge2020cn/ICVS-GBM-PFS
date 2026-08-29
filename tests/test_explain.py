import pandas as pd
import pytest

from icvs_gbm_pfs.explain import _bootstrap_shapley_intervals, _shapley_dependence_curve


def test_shapley_intervals_resample_patients() -> None:
    values = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P2", "P2"],
            "horizon_months": [12.0, 12.0, 12.0, 12.0],
            "feature": ["age", "vit", "age", "vit"],
            "shapley_value": [0.10, -0.20, 0.30, -0.40],
        }
    )
    result = _bootstrap_shapley_intervals(values, "patient_id", resamples=200, seed=2026)
    age = result.loc[result["feature"].eq("age")].iloc[0]
    assert age["mean_absolute_shapley_ci_low"] == pytest.approx(0.10)
    assert age["mean_absolute_shapley_ci_high"] == pytest.approx(0.30)


def test_shapley_dependence_curve_reports_bootstrap_band() -> None:
    values = pd.DataFrame(
        {
            "patient_id": [f"P{index}" for index in range(8)],
            "horizon_months": [12.0] * 8,
            "feature": ["vit_score_standardized"] * 8,
            "feature_value": [-1.5, -1.0, -0.5, -0.2, 0.2, 0.5, 1.0, 1.5],
            "shapley_value": [-0.3, -0.2, -0.1, -0.04, 0.04, 0.1, 0.2, 0.3],
        }
    )
    result = _shapley_dependence_curve(
        values,
        "patient_id",
        feature="vit_score_standardized",
        horizon=12.0,
        resamples=50,
        seed=2026,
        grid_points=12,
    )
    assert len(result) == 12
    assert (result["ci_low"] <= result["ci_high"]).all()
