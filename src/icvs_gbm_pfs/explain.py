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


def _progression_probabilities(
    model: RandomSurvivalForest,
    features: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    survival = model.predict_survival_function(features, return_array=True)
    indices = np.searchsorted(model.unique_times_, horizons, side="right") - 1
    result = np.zeros((len(features), len(horizons)), dtype=float)
    available = indices >= 0
    result[:, available] = 1.0 - survival[:, indices[available]]
    return result


def _feature_matrix(
    frame: pd.DataFrame,
    config: StudyConfig,
    scaler: StandardScaler,
    vit_score_column: str,
) -> np.ndarray:
    vit = scaler.transform(frame[[vit_score_column]]).reshape(-1)
    matrix = np.column_stack(
        [
            frame[config.column("age")].to_numpy(float),
            frame[config.column("mgmt")].to_numpy(float),
            frame[config.column("extent_of_resection")].to_numpy(float),
            vit,
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Explanation predictors must contain only finite values.")
    return matrix


def select_explanation_patients(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    total: int,
) -> pd.DataFrame:
    """Select a deterministic cohort-proportional patient set without using outcomes."""

    if total <= 0:
        raise ValueError("The requested explanation sample size must be greater than zero.")
    if frame.empty:
        raise ValueError("Explanation selection requires at least one patient.")
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


def _bootstrap_shapley_intervals(
    values: pd.DataFrame,
    patient_column: str,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate patient-level bootstrap intervals for mean absolute Shapley values."""

    if resamples <= 0:
        raise ValueError("Shapley bootstrap resamples must be greater than zero.")
    required = {patient_column, "horizon_months", "feature", "shapley_value"}
    missing = sorted(required.difference(values.columns))
    if missing:
        raise ValueError(f"Shapley values are missing columns: {', '.join(missing)}")
    matrix = values.assign(absolute_shapley=values["shapley_value"].abs()).pivot(
        index=patient_column,
        columns=["horizon_months", "feature"],
        values="absolute_shapley",
    )
    if matrix.empty or matrix.isna().any().any():
        raise ValueError("Every explained patient must have one value per horizon and feature.")
    rng = np.random.default_rng(seed)
    observed = matrix.to_numpy(dtype=float)
    bootstrap = np.empty((resamples, observed.shape[1]), dtype=float)
    for index in range(resamples):
        selected = rng.integers(0, observed.shape[0], size=observed.shape[0])
        bootstrap[index] = observed[selected].mean(axis=0)
    intervals = matrix.columns.to_frame(index=False)
    intervals["mean_absolute_shapley_ci_low"] = np.percentile(bootstrap, 2.5, axis=0)
    intervals["mean_absolute_shapley_ci_high"] = np.percentile(bootstrap, 97.5, axis=0)
    return intervals


def _local_linear_smooth(
    feature_values: np.ndarray,
    shapley_values: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    estimates = np.empty(len(grid), dtype=float)
    for index, location in enumerate(grid):
        centered = feature_values - location
        weights = np.exp(-0.5 * np.square(centered / bandwidth))
        design = np.column_stack([np.ones(len(centered)), centered])
        coefficients = np.linalg.pinv(design.T @ (weights[:, None] * design)) @ (
            design.T @ (weights * shapley_values)
        )
        estimates[index] = coefficients[0]
    return estimates


def _shapley_dependence_curve(
    values: pd.DataFrame,
    patient_column: str,
    *,
    feature: str,
    horizon: float,
    resamples: int,
    seed: int,
    grid_points: int = 100,
) -> pd.DataFrame:
    """Estimate a patient-bootstrap local-linear Shapley dependence curve."""

    selected = values.loc[
        values["feature"].eq(feature) & values["horizon_months"].eq(horizon),
        [patient_column, "feature_value", "shapley_value"],
    ].copy()
    if selected.empty or selected[patient_column].duplicated().any():
        raise ValueError("Shapley dependence requires one selected row per patient.")
    feature_values = selected["feature_value"].to_numpy(dtype=float)
    shapley_values = selected["shapley_value"].to_numpy(dtype=float)
    if not np.isfinite(feature_values).all() or not np.isfinite(shapley_values).all():
        raise ValueError("Shapley dependence values must be finite.")
    if len(feature_values) < 3 or grid_points < 2 or resamples <= 0:
        raise ValueError("Shapley dependence requires at least three patients and two grid points.")
    standard_deviation = float(np.std(feature_values, ddof=1))
    bandwidth = 1.06 * standard_deviation * len(feature_values) ** (-0.20)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Shapley dependence requires nonconstant feature values.")
    lower, upper = np.percentile(feature_values, [2.5, 97.5])
    grid = np.linspace(lower, upper, grid_points)
    estimate = _local_linear_smooth(feature_values, shapley_values, grid, bandwidth)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((resamples, grid_points), dtype=float)
    for index in range(resamples):
        sampled = rng.integers(0, len(feature_values), size=len(feature_values))
        bootstrap[index] = _local_linear_smooth(
            feature_values[sampled],
            shapley_values[sampled],
            grid,
            bandwidth,
        )
    return pd.DataFrame(
        {
            "feature": feature,
            "horizon_months": horizon,
            "feature_value": grid,
            "shapley_value_smoothed": estimate,
            "ci_low": np.percentile(bootstrap, 2.5, axis=0),
            "ci_high": np.percentile(bootstrap, 97.5, axis=0),
            "bandwidth": bandwidth,
        }
    )


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
    if background_frame.empty or explanation_frame.empty:
        raise ValueError("Shapley analysis requires nonempty background and explanation cohorts.")
    if not background_frame[config.column("cohort")].eq(config.cohort("training")).all():
        raise ValueError("The Shapley background must contain training-cohort patients only.")
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
        coalition_values: dict[int, np.ndarray] = {}
        for coalition_bits in range(1 << feature_count):
            combined = background.copy()
            for feature_index in range(feature_count):
                if coalition_bits & (1 << feature_index):
                    combined[:, feature_index] = patient_features[feature_index]
            coalition_values[coalition_bits] = _progression_probabilities(
                model,
                combined,
                horizons,
            ).mean(axis=0)
        for horizon_index, horizon in enumerate(horizons):
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
                            coalition_values[bits | (1 << feature_index)][horizon_index]
                            - coalition_values[bits][horizon_index]
                        )
            baseline = float(coalition_values[0][horizon_index])
            prediction = float(coalition_values[(1 << feature_count) - 1][horizon_index])
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
    intervals = _bootstrap_shapley_intervals(
        values,
        patient_col,
        resamples=int(config.section("evaluation")["bootstrap_resamples"]),
        seed=config.seed,
    )
    summary = summary.merge(intervals, on=["horizon_months", "feature"], validate="one_to_one")
    contribution_total = summary.groupby("horizon_months")["mean_absolute_shapley"].transform(
        "sum"
    )
    summary["relative_contribution"] = np.divide(
        summary["mean_absolute_shapley"],
        contribution_total,
        out=np.full(len(summary), np.nan, dtype=float),
        where=contribution_total.to_numpy(float) > 0,
    )
    dependence = _shapley_dependence_curve(
        values,
        patient_col,
        feature="vit_score_standardized",
        horizon=12.0,
        resamples=int(config.section("evaluation")["bootstrap_resamples"]),
        seed=config.seed + 1,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / "icvs_time_dependent_shapley_values.csv", index=False)
    summary.to_csv(output / "icvs_time_dependent_shapley_summary.csv", index=False)
    dependence.to_csv(output / "icvs_vit_12m_dependence.csv", index=False)
    return values, summary
