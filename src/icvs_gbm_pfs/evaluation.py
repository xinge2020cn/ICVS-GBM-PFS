"""Locked-cohort survival discrimination, calibration, and risk-stratification analyses."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
)

from .config import StudyConfig
from .survival import structured_survival


def _metric_values(
    training_outcome: np.ndarray,
    test_outcome: np.ndarray,
    risk_score: np.ndarray,
    survival_probability: np.ndarray,
    horizons: np.ndarray,
) -> dict[str, float]:
    c_index = float(
        concordance_index_censored(test_outcome["event"], test_outcome["time"], risk_score)[0]
    )
    time_auc, integrated_auc = cumulative_dynamic_auc(
        training_outcome,
        test_outcome,
        risk_score,
        horizons,
    )
    brier_times, brier_values = brier_score(
        training_outcome,
        test_outcome,
        survival_probability,
        horizons,
    )
    result = {"c_index": c_index, "integrated_auc": float(integrated_auc)}
    for index, horizon in enumerate(horizons):
        result[f"auc_{int(horizon)}m"] = float(time_auc[index])
        result[f"brier_{int(horizon)}m"] = float(brier_values[index])
        if not np.isclose(brier_times[index], horizon):
            raise RuntimeError("Brier-score horizons changed unexpectedly.")
    result["integrated_brier_score"] = float(
        np.trapezoid(brier_values, brier_times) / (brier_times[-1] - brier_times[0])
    )
    return result


def _calibration_table(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    time_column: str,
    event_column: str,
    horizon: float,
    groups: int,
) -> pd.DataFrame:
    calibration = frame[[probability_column, time_column, event_column]].dropna().copy()
    calibration["group"] = pd.qcut(
        calibration[probability_column], q=groups, labels=False, duplicates="drop"
    )
    rows = []
    for group, values in calibration.groupby("group", sort=True):
        km = KaplanMeierFitter().fit(values[time_column], values[event_column])
        survival = float(km.predict(horizon))
        confidence = km.confidence_interval_survival_function_
        index = confidence.index.searchsorted(horizon, side="right") - 1
        if index >= 0:
            survival_low = float(confidence.iloc[index, 0])
            survival_high = float(confidence.iloc[index, 1])
        else:
            survival_low = survival_high = 1.0
        rows.append(
            {
                "group": int(group) + 1,
                "n": len(values),
                "mean_predicted_progression": float(values[probability_column].mean()),
                "observed_progression": 1.0 - survival,
                "observed_progression_ci_low": 1.0 - survival_high,
                "observed_progression_ci_high": 1.0 - survival_low,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_metric_intervals(
    training_outcome: np.ndarray,
    test_outcome: np.ndarray,
    risk_score: np.ndarray,
    survival_probability: np.ndarray,
    horizons: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {"c_index": [], "integrated_auc": [], "integrated_brier_score": []}
    for _ in range(resamples):
        selected = rng.integers(0, len(test_outcome), size=len(test_outcome))
        outcome = test_outcome[selected]
        try:
            metrics = _metric_values(
                training_outcome,
                outcome,
                risk_score[selected],
                survival_probability[selected],
                horizons,
            )
        except (ValueError, ZeroDivisionError):
            continue
        for metric in values:
            values[metric].append(metrics[metric])
    result = {}
    for metric, estimates in values.items():
        array = np.asarray(estimates, dtype=float)
        if array.size < max(5, resamples // 2):
            raise RuntimeError(f"Too few valid bootstrap resamples were available for {metric}.")
        result[f"{metric}_ci_low"] = float(np.percentile(array, 2.5))
        result[f"{metric}_ci_high"] = float(np.percentile(array, 97.5))
    result["valid_bootstrap_resamples"] = int(min(len(item) for item in values.values()))
    return result


def _risk_stratification(
    frame: pd.DataFrame,
    *,
    score_column: str,
    cutoff: float,
    time_column: str,
    event_column: str,
    adjustment_columns: list[str],
) -> dict[str, float]:
    values = frame[[time_column, event_column, score_column, *adjustment_columns]].copy()
    values["high_risk"] = (values[score_column] > cutoff).astype(int)
    high = values.loc[values["high_risk"].eq(1)]
    low = values.loc[values["high_risk"].eq(0)]
    if high.empty or low.empty:
        raise ValueError("The locked cutoff does not create two risk groups in this cohort.")
    logrank = logrank_test(
        high[time_column],
        low[time_column],
        event_observed_A=high[event_column],
        event_observed_B=low[event_column],
    )
    unadjusted = CoxPHFitter(penalizer=1e-8).fit(
        values[[time_column, event_column, "high_risk"]],
        duration_col=time_column,
        event_col=event_column,
    )
    adjusted = CoxPHFitter(penalizer=1e-8).fit(
        values[[time_column, event_column, "high_risk", *adjustment_columns]],
        duration_col=time_column,
        event_col=event_column,
    )
    unadjusted_row = unadjusted.summary.loc["high_risk"]
    adjusted_row = adjusted.summary.loc["high_risk"]
    return {
        "cutoff": cutoff,
        "low_risk_n": len(low),
        "high_risk_n": len(high),
        "logrank_p_value": float(logrank.p_value),
        "unadjusted_hr": float(unadjusted_row["exp(coef)"]),
        "unadjusted_ci_low": float(unadjusted_row["exp(coef) lower 95%"]),
        "unadjusted_ci_high": float(unadjusted_row["exp(coef) upper 95%"]),
        "adjusted_hr": float(adjusted_row["exp(coef)"]),
        "adjusted_ci_low": float(adjusted_row["exp(coef) lower 95%"]),
        "adjusted_ci_high": float(adjusted_row["exp(coef) upper 95%"]),
        "adjusted_p_value": float(adjusted_row["p"]),
    }


def _bootstrap_pairwise(
    frame: pd.DataFrame,
    training_outcome: np.ndarray,
    first_model: str,
    second_model: str,
    horizons: np.ndarray,
    *,
    resamples: int,
    seed: int,
    time_column: str,
    event_column: str,
) -> list[dict[str, float | str]]:
    first = frame.loc[frame["model"].eq(first_model)].sort_values("patient_id")
    second = frame.loc[frame["model"].eq(second_model)].sort_values("patient_id")
    if not np.array_equal(first["patient_id"].to_numpy(), second["patient_id"].to_numpy()):
        raise ValueError("Pairwise model comparisons require identical patient sets.")
    outcome = structured_survival(first[event_column], first[time_column])
    first_risk = first["risk_score"].to_numpy(float)
    second_risk = second["risk_score"].to_numpy(float)
    rng = np.random.default_rng(seed)
    differences = {"c_index": [], "integrated_auc": []}
    for _ in range(resamples):
        selected = rng.integers(0, len(first), size=len(first))
        sampled_outcome = outcome[selected]
        try:
            first_c = concordance_index_censored(
                sampled_outcome["event"], sampled_outcome["time"], first_risk[selected]
            )[0]
            second_c = concordance_index_censored(
                sampled_outcome["event"], sampled_outcome["time"], second_risk[selected]
            )[0]
            _, first_auc = cumulative_dynamic_auc(
                training_outcome, sampled_outcome, first_risk[selected], horizons
            )
            _, second_auc = cumulative_dynamic_auc(
                training_outcome, sampled_outcome, second_risk[selected], horizons
            )
        except (ValueError, ZeroDivisionError):
            continue
        differences["c_index"].append(float(first_c - second_c))
        differences["integrated_auc"].append(float(first_auc - second_auc))
    rows = []
    for metric, values in differences.items():
        array = np.asarray(values, dtype=float)
        if array.size < max(100, resamples // 2):
            raise RuntimeError("Too few valid paired bootstrap resamples were available.")
        probability = min(np.mean(array <= 0), np.mean(array >= 0))
        p_value = min(1.0, 2.0 * (array.size * probability + 1.0) / (array.size + 1.0))
        rows.append(
            {
                "model_1": first_model,
                "model_2": second_model,
                "metric": metric,
                "difference": float(np.mean(array)),
                "ci_low": float(np.percentile(array, 2.5)),
                "ci_high": float(np.percentile(array, 97.5)),
                "p_value": float(p_value),
                "valid_resamples": int(array.size),
            }
        )
    return rows


def evaluate_models(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Evaluate all models using one locked patient-level prediction table."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    horizons = np.asarray(config.section("icvs")["horizons_months"], dtype=float)
    required_prediction_columns = [
        patient_col,
        "model",
        "risk_score",
        *[f"pfs_{int(horizon)}m" for horizon in horizons],
    ]
    missing = [column for column in required_prediction_columns if column not in predictions]
    if missing:
        raise ValueError(f"Prediction table is missing columns: {', '.join(missing)}")
    if predictions.duplicated([patient_col, "model"]).any():
        raise ValueError("Prediction rows must be unique by patient and model.")
    frame = predictions.merge(manifest, on=patient_col, how="left", validate="many_to_one")
    if frame[[cohort_col, time_col, event_col]].isna().any().any():
        raise ValueError("Predictions contain patients absent from the manifest.")
    training_manifest = manifest.loc[manifest[cohort_col].eq(config.cohort("training"))].copy()
    training_outcome = structured_survival(
        training_manifest[event_col], training_manifest[time_col]
    )
    cohort_definitions: list[tuple[str, pd.Series, bool]] = []
    for name in ("training", "temporal_validation", "spatial_validation"):
        cohort_definitions.append((name, frame[cohort_col].eq(config.cohort(name)), True))
    subset_col = config.column("biological_subset")
    has_biological_subset = (
        subset_col in frame
        and pd.to_numeric(frame[subset_col], errors="coerce").fillna(0).eq(1).any()
    )
    if has_biological_subset:
        cohort_definitions.append(
            (
                "biological_subset",
                pd.to_numeric(frame[subset_col], errors="coerce").fillna(0).eq(1),
                False,
            )
        )
    performance_rows = []
    calibration_rows = []
    risk_rows = []
    pairwise_rows = []
    settings = config.section("evaluation")
    adjustment_columns = [
        config.column("age"),
        config.column("mgmt"),
        config.column("extent_of_resection"),
    ]
    cutoffs = (
        frame.loc[frame[cohort_col].eq(config.cohort("training"))]
        .groupby("model")["risk_score"]
        .median()
        .to_dict()
    )
    analysis_index = 0
    for cohort_name, cohort_mask, independent in cohort_definitions:
        cohort = frame.loc[cohort_mask].copy()
        for model_name, model_frame in cohort.groupby("model", sort=True):
            outcome = structured_survival(model_frame[event_col], model_frame[time_col])
            survival_probability = model_frame[
                [f"pfs_{int(horizon)}m" for horizon in horizons]
            ].to_numpy(float)
            values = _metric_values(
                training_outcome,
                outcome,
                model_frame["risk_score"].to_numpy(float),
                survival_probability,
                horizons,
            )
            intervals = _bootstrap_metric_intervals(
                training_outcome,
                outcome,
                model_frame["risk_score"].to_numpy(float),
                survival_probability,
                horizons,
                resamples=int(settings["bootstrap_resamples"]),
                seed=config.seed + analysis_index,
            )
            analysis_index += 1
            performance_rows.append(
                {
                    "cohort": cohort_name,
                    "independent_validation": independent and cohort_name != "training",
                    "model": model_name,
                    "n": len(model_frame),
                    "events": int(model_frame[event_col].sum()),
                    **values,
                    **intervals,
                }
            )
            calibration_horizon = float(settings["calibration_horizon_months"])
            probability_column = f"pfs_{int(calibration_horizon)}m"
            calibration_input = model_frame.copy()
            calibration_input["progression_probability"] = (
                1.0 - calibration_input[probability_column]
            )
            calibration = _calibration_table(
                calibration_input,
                probability_column="progression_probability",
                time_column=time_col,
                event_column=event_col,
                horizon=calibration_horizon,
                groups=int(settings["calibration_groups"]),
            )
            calibration.insert(0, "model", model_name)
            calibration.insert(0, "cohort", cohort_name)
            calibration_rows.extend(calibration.to_dict(orient="records"))
            risk_rows.append(
                {
                    "cohort": cohort_name,
                    "model": model_name,
                    **_risk_stratification(
                        model_frame,
                        score_column="risk_score",
                        cutoff=float(cutoffs[model_name]),
                        time_column=time_col,
                        event_column=event_col,
                        adjustment_columns=adjustment_columns,
                    ),
                }
            )
        models = sorted(cohort["model"].unique())
        for pair_index, (first_model, second_model) in enumerate(itertools.combinations(models, 2)):
            rows = _bootstrap_pairwise(
                cohort.rename(columns={patient_col: "patient_id"}),
                training_outcome,
                first_model,
                second_model,
                horizons,
                resamples=int(settings["bootstrap_resamples"]),
                seed=config.seed + pair_index,
                time_column=time_col,
                event_column=event_col,
            )
            for row in rows:
                row["cohort"] = cohort_name
                row["independent_validation"] = independent and cohort_name != "training"
            pairwise_rows.extend(rows)
    tables = {
        "performance": pd.DataFrame(performance_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "risk_stratification": pd.DataFrame(risk_rows),
        "pairwise_comparisons": pd.DataFrame(pairwise_rows),
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "evaluation_metadata.json").write_text(
        json.dumps(
            {
                "horizons_months": horizons.tolist(),
                "bootstrap_resamples": int(settings["bootstrap_resamples"]),
                "training_censoring_distribution_used_for_all_cohorts": True,
                "biological_subset_is_independent": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tables
