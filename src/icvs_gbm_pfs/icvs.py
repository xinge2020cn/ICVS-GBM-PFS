"""Integrated Clinical-ViT Survival model fitting and locked inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest

from .config import StudyConfig
from .survival import structured_survival


def _survival_at_horizons(
    model: RandomSurvivalForest,
    features: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    survival = model.predict_survival_function(features, return_array=True)
    indices = np.searchsorted(model.unique_times_, horizons, side="right") - 1
    result = np.ones((len(features), len(horizons)), dtype=float)
    available = indices >= 0
    result[:, available] = survival[:, indices[available]]
    return result


def _oob_survival_at_horizons(
    model: RandomSurvivalForest,
    training_features: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """Aggregate per-tree survival functions only for out-of-bag patients."""

    total = np.zeros((len(training_features), len(horizons)), dtype=float)
    count = np.zeros(len(training_features), dtype=int)
    for tree, in_bag_indices in zip(model.estimators_, model.estimators_samples_, strict=True):
        is_oob = np.ones(len(training_features), dtype=bool)
        is_oob[np.unique(in_bag_indices)] = False
        if not is_oob.any():
            continue
        tree_survival = tree.predict_survival_function(training_features[is_oob], return_array=True)
        indices = np.searchsorted(tree.unique_times_, horizons, side="right") - 1
        values = np.ones((is_oob.sum(), len(horizons)), dtype=float)
        available = indices >= 0
        values[:, available] = tree_survival[:, indices[available]]
        total[is_oob] += values
        count[is_oob] += 1
    if np.any(count == 0):
        raise RuntimeError("Out-of-bag survival estimates are unavailable for some patients.")
    return total / count[:, None]


def _build_feature_matrix(
    frame: pd.DataFrame,
    config: StudyConfig,
    standardized_vit_score: np.ndarray,
) -> np.ndarray:
    age = pd.to_numeric(frame[config.column("age")], errors="raise").to_numpy(float)
    mgmt = pd.to_numeric(frame[config.column("mgmt")], errors="raise").to_numpy(float)
    non_gtr = pd.to_numeric(frame[config.column("extent_of_resection")], errors="raise").to_numpy(
        float
    )
    for name, values in (("MGMT", mgmt), ("extent of resection", non_gtr)):
        if not set(np.unique(values)).issubset({0.0, 1.0}):
            raise ValueError(f"{name} must use binary zero-one coding.")
    matrix = np.column_stack([age, mgmt, non_gtr, standardized_vit_score]).astype(float)
    if not np.isfinite(matrix).all():
        raise ValueError("ICVS predictors contain missing or nonfinite values.")
    return matrix


def fit_icvs_model(
    manifest: pd.DataFrame,
    vit_scores: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Fit the locked random survival forest with OOF development predictions."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    required_scores = [patient_col, "vit_score_oof", "vit_score_final"]
    missing = [column for column in required_scores if column not in vit_scores]
    if missing:
        raise ValueError(f"ViT score table is missing columns: {', '.join(missing)}")
    if vit_scores[patient_col].duplicated().any():
        raise ValueError("ViT score table contains duplicate patient identifiers.")
    if set(manifest[patient_col].astype(str)) != set(vit_scores[patient_col].astype(str)):
        raise ValueError("ViT scores must contain exactly one row for every patient.")
    frame = manifest.merge(
        vit_scores[required_scores], on=patient_col, how="left", validate="one_to_one"
    )
    if not np.isfinite(frame["vit_score_final"].to_numpy(float)).all():
        raise ValueError("Final ViT scores must be finite for every patient.")
    training_mask = frame[cohort_col].eq(config.cohort("training")).to_numpy()
    training = frame.loc[training_mask].reset_index(drop=True)
    if not np.isfinite(training["vit_score_oof"].to_numpy(float)).all():
        raise ValueError("Every training patient must have one finite out-of-fold ViT score.")
    oof_scaler = StandardScaler().fit(training[["vit_score_oof"]])
    oof_standardized = oof_scaler.transform(training[["vit_score_oof"]]).reshape(-1)
    final_scaler = StandardScaler().fit(training[["vit_score_final"]])
    final_standardized = final_scaler.transform(frame[["vit_score_final"]]).reshape(-1)
    training_features = _build_feature_matrix(training, config, oof_standardized)
    all_features = _build_feature_matrix(frame, config, final_standardized)
    outcome = structured_survival(training[event_col], training[time_col])
    settings = config.section("icvs")
    model = RandomSurvivalForest(
        n_estimators=int(settings["n_estimators"]),
        max_features=int(settings["max_features"]),
        min_samples_split=int(settings["min_samples_split"]),
        min_samples_leaf=int(settings["min_samples_leaf"]),
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=config.seed,
    )
    model.fit(training_features, outcome)
    risk_score = model.predict(all_features)
    if not np.isfinite(risk_score).all() or not np.isfinite(model.oob_prediction_).all():
        raise RuntimeError("ICVS fitting produced nonfinite risk predictions.")
    risk_score[training_mask] = model.oob_prediction_
    horizons = np.asarray(settings["horizons_months"], dtype=float)
    if (
        horizons.ndim != 1
        or horizons.size == 0
        or not np.isfinite(horizons).all()
        or np.any(horizons <= 0)
        or np.any(np.diff(horizons) <= 0)
    ):
        raise ValueError("ICVS horizons must be strictly increasing positive values.")
    survival = _survival_at_horizons(model, all_features, horizons)
    survival[training_mask] = _oob_survival_at_horizons(model, training_features, horizons)
    if not np.isfinite(survival).all() or np.any((survival < 0.0) | (survival > 1.0)):
        raise RuntimeError("ICVS fitting produced invalid survival probabilities.")
    cutoff = float(np.median(model.oob_prediction_))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = frame[[patient_col, cohort_col]].copy()
    predictions["icvs_risk_score"] = risk_score
    predictions["icvs_risk_group"] = np.where(risk_score > cutoff, "high", "low")
    predictions["icvs_training_cutoff"] = cutoff
    for horizon_index, horizon in enumerate(horizons):
        predictions[f"icvs_pfs_{int(horizon)}m"] = survival[:, horizon_index]
    predictions.to_csv(output / "icvs_predictions.csv", index=False)
    artifact = {
        "model": model,
        "feature_order": [
            config.column("age"),
            config.column("mgmt"),
            config.column("extent_of_resection"),
            "vit_score_standardized",
        ],
        "oof_vit_scaler": oof_scaler,
        "final_vit_scaler": final_scaler,
        "training_cutoff": cutoff,
        "horizons_months": horizons,
        "training_patient_ids": training[patient_col].astype(str).tolist(),
    }
    joblib.dump(artifact, output / "icvs_model.joblib")
    (output / "icvs_model_metadata.json").write_text(
        json.dumps(
            {
                "algorithm": "random_survival_forest",
                "n_estimators": int(settings["n_estimators"]),
                "max_features": int(settings["max_features"]),
                "min_samples_split": int(settings["min_samples_split"]),
                "min_samples_leaf": int(settings["min_samples_leaf"]),
                "training_cutoff": cutoff,
                "horizons_months": horizons.tolist(),
                "training_predictions": "out_of_bag",
                "validation_predictions": "locked_full_forest",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return predictions


def predict_icvs(
    artifact_path: str | Path,
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    vit_score_column: str = "vit_score_final",
) -> pd.DataFrame:
    """Apply the locked ICVS artifact without refitting or cutoff selection."""

    artifact = joblib.load(artifact_path)
    scaler: StandardScaler = artifact["final_vit_scaler"]
    standardized = scaler.transform(frame[[vit_score_column]]).reshape(-1)
    features = _build_feature_matrix(frame, config, standardized)
    model: RandomSurvivalForest = artifact["model"]
    horizons = np.asarray(artifact["horizons_months"], dtype=float)
    risk = model.predict(features)
    survival = _survival_at_horizons(model, features, horizons)
    result = frame[[config.column("patient_id")]].copy()
    result["icvs_risk_score"] = risk
    result["icvs_risk_group"] = np.where(risk > float(artifact["training_cutoff"]), "high", "low")
    for horizon_index, horizon in enumerate(horizons):
        result[f"icvs_pfs_{int(horizon)}m"] = survival[:, horizon_index]
    return result
