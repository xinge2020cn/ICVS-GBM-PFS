"""Locked clinical Cox comparator for progression-free survival."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .config import StudyConfig
from .survival import structured_survival


def _clinical_matrix(frame: pd.DataFrame, config: StudyConfig) -> np.ndarray:
    age_per_decade = frame[config.column("age")].to_numpy(float) / 10.0
    mgmt = frame[config.column("mgmt")].to_numpy(float)
    non_gtr = frame[config.column("extent_of_resection")].to_numpy(float)
    matrix = np.column_stack([age_per_decade, mgmt, non_gtr])
    if not np.isfinite(matrix).all():
        raise ValueError("Clinical predictors contain missing or nonfinite values.")
    if not set(np.unique(mgmt)).issubset({0.0, 1.0}):
        raise ValueError("MGMT promoter methylation must use binary zero-one coding.")
    if not set(np.unique(non_gtr)).issubset({0.0, 1.0}):
        raise ValueError("Extent of resection must use binary zero-one coding.")
    return matrix


def fit_clinical_model(
    manifest: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Fit the prespecified retained clinical predictors in the training cohort."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    training_mask = manifest[cohort_col].eq(config.cohort("training")).to_numpy()
    training = manifest.loc[training_mask]
    training_features = _clinical_matrix(training, config)
    all_features = _clinical_matrix(manifest, config)
    outcome = structured_survival(
        training[config.column("pfs_event")], training[config.column("pfs_time")]
    )
    model = CoxPHSurvivalAnalysis(alpha=1e-8, ties="breslow").fit(training_features, outcome)
    risk = model.predict(all_features)
    horizons = np.asarray(config.section("icvs")["horizons_months"], dtype=float)
    functions = model.predict_survival_function(all_features)
    survival = np.column_stack(
        [[float(function(horizon)) for function in functions] for horizon in horizons]
    )
    cutoff = float(np.median(risk[training_mask]))
    result = manifest[[patient_col, cohort_col]].copy()
    result["clinical_risk_score"] = risk
    result["clinical_risk_group"] = np.where(risk > cutoff, "high", "low")
    result["clinical_training_cutoff"] = cutoff
    for horizon_index, horizon in enumerate(horizons):
        result[f"clinical_pfs_{int(horizon)}m"] = survival[:, horizon_index]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "clinical_predictions.csv", index=False)
    joblib.dump(
        {
            "model": model,
            "feature_order": [
                "age_per_decade",
                config.column("mgmt"),
                config.column("extent_of_resection"),
            ],
            "training_cutoff": cutoff,
            "horizons_months": horizons,
            "training_patient_ids": training[patient_col].astype(str).tolist(),
        },
        output / "clinical_model.joblib",
    )
    (output / "clinical_model_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "cox_proportional_hazards",
                "ties": "breslow",
                "predictors": [
                    "age_per_decade",
                    config.column("mgmt"),
                    config.column("extent_of_resection"),
                ],
                "coefficients": model.coef_.tolist(),
                "training_cutoff": cutoff,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
