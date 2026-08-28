"""Exact coalition-based time-dependent Shapley analysis for the four-predictor ICVS."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest

from .config import StudyConfig


def _progression_probability(
    model: RandomSurvivalForest,
    features: np.ndarray,
    horizon: float,
) -> np.ndarray:
    survival = model.predict_survival_function(features, return_array=True)
    index = int(np.searchsorted(model.unique_times_, horizon, side="right") - 1)
    if index < 0:
        return np.zeros(len(features), dtype=float)
    return 1.0 - survival[:, index]


def _feature_matrix(
    frame: pd.DataFrame,
    config: StudyConfig,
    scaler: StandardScaler,
    vit_score_column: str,
) -> np.ndarray:
    vit = scaler.transform(frame[[vit_score_column]]).reshape(-1)
    return np.column_stack(
        [
            frame[config.column("age")].to_numpy(float),
            frame[config.column("mgmt")].to_numpy(float),
            frame[config.column("extent_of_resection")].to_numpy(float),
            vit,
        ]
    )


def select_explanation_patients(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    total: int,
) -> pd.DataFrame:
    """Select a deterministic cohort-proportional patient set without using outcomes."""

    if total >= len(frame):
        return frame.copy().reset_index(drop=True)
    cohort_col = config.column("cohort")
    rng = np.random.default_rng(config.seed)
    allocations = (
        frame[cohort_col].value_counts(normalize=True).mul(total).round().astype(int).to_dict()
    )
    difference = total - sum(allocations.values())
    largest = frame[cohort_col].value_counts().index[0]
    allocations[largest] += difference
    selected = []
    for cohort, count in allocations.items():
        indices = frame.index[frame[cohort_col].eq(cohort)].to_numpy()
        selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    return frame.loc[selected].sort_index().reset_index(drop=True)


def exact_time_dependent_shapley(
    artifact_path: str | Path,
    background_frame: pd.DataFrame,
    explanation_frame: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
    *,
    final_vit_score_column: str = "vit_score_final",
    oof_vit_score_column: str = "vit_score_oof",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate exact four-feature Shapley values against the training background."""

    artifact = joblib.load(artifact_path)
    model: RandomSurvivalForest = artifact["model"]
    final_scaler: StandardScaler = artifact["final_vit_scaler"]
    oof_scaler: StandardScaler = artifact["oof_vit_scaler"]
    feature_names = list(artifact["feature_order"])
    if len(feature_names) != 4:
        raise ValueError("Exact ICVS Shapley analysis requires exactly four predictors.")
    if background_frame[oof_vit_score_column].isna().any():
        raise ValueError("Training background rows require out-of-fold ViT scores.")
    background = _feature_matrix(
        background_frame,
        config,
        oof_scaler,
        oof_vit_score_column,
    )
    explained = _feature_matrix(
        explanation_frame,
        config,
        final_scaler,
        final_vit_score_column,
    )
    training_mask = explanation_frame[config.column("cohort")].eq(config.cohort("training"))
    if explanation_frame.loc[training_mask, oof_vit_score_column].isna().any():
        raise ValueError("Explained training rows require out-of-fold ViT scores.")
    explained[training_mask.to_numpy()] = _feature_matrix(
        explanation_frame.loc[training_mask],
        config,
        oof_scaler,
        oof_vit_score_column,
    )
    horizons = np.asarray(artifact["horizons_months"], dtype=float)
    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    feature_count = len(feature_names)
    factorial = math.factorial
    rows = []
    for patient_index, patient_features in enumerate(explained):
        for horizon in horizons:
            coalition_values: dict[int, float] = {}
            for coalition_bits in range(1 << feature_count):
                combined = background.copy()
                for feature_index in range(feature_count):
                    if coalition_bits & (1 << feature_index):
                        combined[:, feature_index] = patient_features[feature_index]
                coalition_values[coalition_bits] = float(
                    _progression_probability(model, combined, float(horizon)).mean()
                )
            shapley_values = np.zeros(feature_count, dtype=float)
            for feature_index in range(feature_count):
                for subset_size in range(feature_count):
                    for subset in itertools.combinations(
                        [index for index in range(feature_count) if index != feature_index],
                        subset_size,
                    ):
                        bits = sum(1 << index for index in subset)
                        weight = (
                            factorial(subset_size)
                            * factorial(feature_count - subset_size - 1)
                            / factorial(feature_count)
                        )
                        shapley_values[feature_index] += weight * (
                            coalition_values[bits | (1 << feature_index)] - coalition_values[bits]
                        )
            baseline = coalition_values[0]
            prediction = coalition_values[(1 << feature_count) - 1]
            additivity_error = float(abs(baseline + shapley_values.sum() - prediction))
            if additivity_error > 1e-8:
                raise RuntimeError(
                    f"Shapley additivity check failed with error {additivity_error:.3e}."
                )
            for feature_index, feature_name in enumerate(feature_names):
                rows.append(
                    {
                        patient_col: str(explanation_frame.iloc[patient_index][patient_col]),
                        cohort_col: str(explanation_frame.iloc[patient_index][cohort_col]),
                        "horizon_months": float(horizon),
                        "feature": feature_name,
                        "feature_value": float(patient_features[feature_index]),
                        "shapley_value": float(shapley_values[feature_index]),
                        "baseline_progression_probability": baseline,
                        "predicted_progression_probability": prediction,
                        "additivity_error": additivity_error,
                    }
                )
    values = pd.DataFrame(rows)
    summary = (
        values.assign(absolute_shapley=lambda table: table["shapley_value"].abs())
        .groupby(["horizon_months", "feature"], as_index=False)
        .agg(mean_absolute_shapley=("absolute_shapley", "mean"))
    )
    summary["relative_contribution"] = summary["mean_absolute_shapley"] / summary.groupby(
        "horizon_months"
    )["mean_absolute_shapley"].transform("sum")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / "icvs_time_dependent_shapley_values.csv", index=False)
    summary.to_csv(output / "icvs_time_dependent_shapley_summary.csv", index=False)
    return values, summary
