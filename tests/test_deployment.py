from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from deployment.predictor import ICVSPredictor


class _SurvivalModel:
    unique_times_ = np.array([6.0, 12.0, 18.0, 24.0, 30.0, 36.0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (1, 4)
        return np.array([4.0])

    def predict_survival_function(
        self,
        features: np.ndarray,
        *,
        return_array: bool,
    ) -> np.ndarray:
        assert features.shape == (1, 4)
        assert return_array
        return np.array([[0.90, 0.75, 0.62, 0.50, 0.41, 0.32]])


def _bundle() -> dict[str, object]:
    return {
        "model": _SurvivalModel(),
        "feature_order": [
            "age_years",
            "mgmt_methylated",
            "non_gross_total_resection",
            "vit_score_standardized",
        ],
        "training_cutoff": 3.0,
        "horizons_months": np.array([6, 12, 18, 24, 30, 36]),
    }


def test_predictor_validates_and_returns_locked_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "icvs_model.joblib"
    artifact.write_bytes(b"governed-artifact")
    monkeypatch.setattr("deployment.predictor.joblib.load", lambda _: _bundle())
    predictor = ICVSPredictor(artifact)

    result = predictor.predict(
        {
            "age_years": 60.0,
            "mgmt_promoter_methylated": True,
            "extent_of_resection": "gross_total",
            "vit_score_standardized": 0.25,
        }
    )

    assert result["risk_group"] == "high"
    assert result["pfs_probability_12m"] == pytest.approx(0.75)


def test_predictor_rejects_nonfinite_standardized_vit_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "icvs_model.joblib"
    artifact.write_bytes(b"governed-artifact")
    monkeypatch.setattr("deployment.predictor.joblib.load", lambda _: _bundle())
    predictor = ICVSPredictor(artifact)

    with pytest.raises(ValueError, match="must be finite"):
        predictor.predict(
            {
                "age_years": 60.0,
                "mgmt_promoter_methylated": True,
                "extent_of_resection": "gross_total",
                "vit_score_standardized": float("nan"),
            }
        )


def test_fastapi_prediction_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "icvs_model.joblib"
    artifact.write_bytes(b"governed-artifact")
    monkeypatch.setenv("ICVS_MODEL_ARTIFACT", str(artifact))
    monkeypatch.setattr("deployment.predictor.joblib.load", lambda _: _bundle())
    sys.modules.pop("deployment.app", None)
    application = importlib.import_module("deployment.app")
    request = application.ICVSRequest(
        age_years=60.0,
        mgmt_promoter_methylated=True,
        extent_of_resection="gross_total",
        vit_score_standardized=0.25,
    )
    response = application.predict(request)

    assert response["pfs_probability_12m"] == pytest.approx(0.75)
    assert application.health() == {"status": "ok"}
