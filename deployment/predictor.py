"""Validated inference wrapper for a locked ICVS model artifact."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ICVSPredictor:
    """Apply a fitted ICVS random survival forest without refitting."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        configured = model_path or os.environ.get("ICVS_MODEL_ARTIFACT")
        if configured is None:
            raise RuntimeError("ICVS_MODEL_ARTIFACT must identify a fitted ICVS artifact.")
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ICVS model artifact not found: {path}")
        bundle = joblib.load(path)
        required = {"model", "feature_order", "training_cutoff", "horizons_months"}
        missing = sorted(required.difference(bundle))
        if missing:
            raise ValueError(f"ICVS model artifact is missing keys: {', '.join(missing)}")
        expected_order = [
            "age_years",
            "mgmt_methylated",
            "non_gross_total_resection",
            "vit_score_standardized",
        ]
        if list(bundle["feature_order"]) != expected_order:
            raise ValueError("ICVS model artifact has an unsupported feature order.")
        self.model = bundle["model"]
        self.feature_order = expected_order
        self.training_cutoff = float(bundle["training_cutoff"])
        self.horizons = np.asarray(bundle["horizons_months"], dtype=float)
        if (
            self.horizons.ndim != 1
            or self.horizons.size == 0
            or not np.isfinite(self.horizons).all()
            or np.any(self.horizons <= 0)
            or np.any(np.diff(self.horizons) <= 0)
        ):
            raise ValueError("ICVS model artifact contains invalid prediction horizons.")

    @staticmethod
    def _vectorize(payload: dict[str, Any]) -> np.ndarray:
        age = float(payload["age_years"])
        if not np.isfinite(age) or not 18.0 <= age <= 100.0:
            raise ValueError("age_years must be finite and between 18 and 100.")
        mgmt_value = payload["mgmt_promoter_methylated"]
        if not isinstance(mgmt_value, bool):
            raise TypeError("mgmt_promoter_methylated must be Boolean.")
        extent = str(payload["extent_of_resection"]).strip()
        if extent not in {"gross_total", "non_gross_total"}:
            raise ValueError(
                "extent_of_resection must be gross_total or non_gross_total."
            )
        vit_score = float(payload["vit_score_standardized"])
        if not np.isfinite(vit_score):
            raise ValueError("vit_score_standardized must be finite.")
        return np.array(
            [
                [
                    age,
                    float(mgmt_value),
                    float(extent == "non_gross_total"),
                    vit_score,
                ]
            ],
            dtype=float,
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, float | str]:
        """Return locked risk and PFS probabilities for one validated request."""

        features = self._vectorize(payload)
        risk_score = float(self.model.predict(features)[0])
        survival = self.model.predict_survival_function(features, return_array=True)[0]
        indices = np.searchsorted(self.model.unique_times_, self.horizons, side="right") - 1
        probabilities = np.ones(len(self.horizons), dtype=float)
        available = indices >= 0
        probabilities[available] = survival[indices[available]]
        if not np.isfinite(risk_score) or not np.isfinite(probabilities).all():
            raise RuntimeError("ICVS inference produced nonfinite values.")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise RuntimeError("ICVS inference produced invalid survival probabilities.")
        output: dict[str, float | str] = {
            "risk_score": risk_score,
            "risk_group": "high" if risk_score > self.training_cutoff else "low",
        }
        for month, probability in zip(self.horizons, probabilities, strict=True):
            output[f"pfs_probability_{int(month)}m"] = float(probability)
        return output
