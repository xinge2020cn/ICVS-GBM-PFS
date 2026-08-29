"""Locked-cohort survival discrimination, calibration, and risk-stratification analyses."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
)
from sksurv.nonparametric import CensoringDistributionEstimator

from .config import StudyConfig
from .survival import structured_survival


def _benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    """Adjust one prespecified family of P values while preserving row order."""

    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("P values must be finite and lie between zero and one.")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _validate_prediction_table(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    patient_column: str,
    horizons: np.ndarray,
) -> None:
    """Validate complete patient-model coverage and survival prediction invariants."""

    if predictions.empty:
        raise ValueError("Prediction table contains no rows.")
    model_names = predictions["model"].astype("string")
    if model_names.isna().any() or model_names.str.strip().eq("").any():
        raise ValueError("Every prediction row must contain a nonempty model name.")
    expected_patients = set(manifest[patient_column].astype(str))
    for model_name, group in predictions.groupby("model", sort=False):
        observed_patients = set(group[patient_column].astype(str))
        missing = expected_patients.difference(observed_patients)
        extra = observed_patients.difference(expected_patients)
        if missing or extra:
            raise ValueError(
                f"Model '{model_name}' does not contain exactly one prediction for every patient."
            )
    probability_columns = [f"pfs_{int(horizon)}m" for horizon in horizons]
    risk = pd.to_numeric(predictions["risk_score"], errors="coerce").to_numpy(float)
    probability = predictions[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(risk).all() or not np.isfinite(probability).all():
        raise ValueError("Risk scores and survival probabilities must be finite.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Survival probabilities must lie between zero and one.")
    if np.any(np.diff(probability, axis=1) > 1e-8):
        raise ValueError("Survival probabilities must not increase across prediction horizons.")


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


def _time_dependent_roc_table(
    training_outcome: np.ndarray,
    test_outcome: np.ndarray,
    risk_score: np.ndarray,
    horizons: np.ndarray,
) -> pd.DataFrame:
    """Calculate cumulative/dynamic ROC coordinates with training-based IPCW."""

    risk = np.asarray(risk_score, dtype=float)
    if risk.ndim != 1 or len(risk) != len(test_outcome) or not np.isfinite(risk).all():
        raise ValueError("ROC risk scores must be finite and aligned with the test outcomes.")
    censoring = CensoringDistributionEstimator().fit(training_outcome)
    inverse_censoring_weights = censoring.predict_ipcw(test_outcome)
    rows = []
    unique_risk = np.sort(np.unique(risk))[::-1]
    thresholds = np.concatenate(
        (
            [np.nextafter(unique_risk[0], np.inf)],
            unique_risk,
            [np.nextafter(unique_risk[-1], -np.inf)],
        )
    )
    for horizon in horizons:
        cases = test_outcome["event"] & (test_outcome["time"] <= horizon)
        controls = test_outcome["time"] > horizon
        case_weight_total = float(inverse_censoring_weights[cases].sum())
        if case_weight_total <= 0 or not controls.any():
            raise ValueError(f"ROC analysis is not estimable at {horizon:g} months.")
        positive = risk[:, None] >= thresholds[None, :]
        sensitivity = (
            inverse_censoring_weights[cases, None] * positive[cases]
        ).sum(axis=0) / case_weight_total
        false_positive_rate = positive[controls].mean(axis=0)
        rows.extend(
            {
                "horizon_months": float(horizon),
                "threshold": float(threshold),
                "false_positive_rate": float(false_positive),
                "sensitivity": float(true_positive),
                "specificity": float(1.0 - false_positive),
            }
            for threshold, false_positive, true_positive in zip(
                thresholds, false_positive_rate, sensitivity, strict=True
            )
        )
    return pd.DataFrame(rows)


def _kaplan_meier_curve_table(
    frame: pd.DataFrame,
    *,
    score_column: str,
    cutoff: float,
    time_column: str,
    event_column: str,
) -> pd.DataFrame:
    """Return survival, pointwise intervals, and numbers at risk for locked risk groups."""

    values = frame[[time_column, event_column, score_column]].copy()
    values["risk_group"] = np.where(values[score_column] > cutoff, "high", "low")
    rows = []
    for risk_group, group in values.groupby("risk_group", sort=True):
        estimator = KaplanMeierFitter(alpha=0.05).fit(
            group[time_column],
            event_observed=group[event_column],
            label=str(risk_group),
        )
        curve = estimator.survival_function_
        interval = estimator.confidence_interval_survival_function_
        event_table = estimator.event_table
        for index, time in enumerate(curve.index.to_numpy(dtype=float)):
            rows.append(
                {
                    "risk_group": str(risk_group),
                    "time_months": float(time),
                    "survival_probability": float(curve.iloc[index, 0]),
                    "ci_low": float(interval.iloc[index, 0]),
                    "ci_high": float(interval.iloc[index, 1]),
                    "at_risk": int(event_table.iloc[index]["at_risk"]),
                    "events": int(event_table.iloc[index]["observed"]),
                    "censored": int(event_table.iloc[index]["censored"]),
                }
            )
    return pd.DataFrame(rows)


def _calibration_table(
    frame: pd.DataFrame,
    *,
    progression_probability_column: str,
    time_column: str,
    event_column: str,
    horizon: float,
    groups: int,
) -> pd.DataFrame:
    calibration = frame[
        [progression_probability_column, time_column, event_column]
    ].dropna().copy()
    calibration["group"] = pd.qcut(
        calibration[progression_probability_column],
        q=groups,
        labels=False,
        duplicates="drop",
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
                "mean_predicted_progression": float(
                    values[progression_probability_column].mean()
                ),
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
    metric_names = ["c_index", "integrated_auc", "integrated_brier_score"]
    metric_names.extend(f"auc_{int(horizon)}m" for horizon in horizons)
    metric_names.extend(f"brier_{int(horizon)}m" for horizon in horizons)
    values: dict[str, list[float]] = {metric: [] for metric in metric_names}
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
            if np.isfinite(metrics[metric]):
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
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
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
    unadjusted_ph = proportional_hazard_test(
        unadjusted,
        values[[time_column, event_column, "high_risk"]],
        time_transform="rank",
    ).summary
    adjusted_ph = proportional_hazard_test(
        adjusted,
        values[[time_column, event_column, "high_risk", *adjustment_columns]],
        time_transform="rank",
    ).summary
    unadjusted_row = unadjusted.summary.loc["high_risk"]
    adjusted_row = adjusted.summary.loc["high_risk"]
    summary = {
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
        "unadjusted_ph_p_value": float(unadjusted_ph.loc["high_risk", "p"]),
        "adjusted_ph_p_value": float(adjusted_ph.loc["high_risk", "p"]),
        "adjusted_minimum_ph_p_value": float(adjusted_ph["p"].min()),
    }
    coefficient_rows = []
    for analysis, model, ph_table in (
        ("unadjusted", unadjusted, unadjusted_ph),
        ("adjusted", adjusted, adjusted_ph),
    ):
        for term, row in model.summary.iterrows():
            coefficient_rows.append(
                {
                    "analysis": analysis,
                    "term": str(term),
                    "coefficient": float(row["coef"]),
                    "standard_error": float(row["se(coef)"]),
                    "hazard_ratio": float(row["exp(coef)"]),
                    "ci_low": float(row["exp(coef) lower 95%"]),
                    "ci_high": float(row["exp(coef) upper 95%"]),
                    "p_value": float(row["p"]),
                    "proportional_hazards_p_value": float(ph_table.loc[term, "p"]),
                }
            )
    return summary, coefficient_rows


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
    first_observed = _metric_values(
        training_outcome,
        outcome,
        first_risk,
        first[[f"pfs_{int(horizon)}m" for horizon in horizons]].to_numpy(float),
        horizons,
    )
    second_observed = _metric_values(
        training_outcome,
        outcome,
        second_risk,
        second[[f"pfs_{int(horizon)}m" for horizon in horizons]].to_numpy(float),
        horizons,
    )
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
        array = array[np.isfinite(array)]
        if array.size < max(100, resamples // 2):
            raise RuntimeError("Too few valid paired bootstrap resamples were available.")
        probability = min(np.mean(array <= 0), np.mean(array >= 0))
        p_value = min(1.0, 2.0 * (array.size * probability + 1.0) / (array.size + 1.0))
        rows.append(
            {
                "model_1": first_model,
                "model_2": second_model,
                "metric": metric,
                "difference": float(first_observed[metric] - second_observed[metric]),
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
    if (
        horizons.ndim != 1
        or horizons.size < 2
        or not np.isfinite(horizons).all()
        or np.any(horizons <= 0)
        or np.any(np.diff(horizons) <= 0)
    ):
        raise ValueError(
            "Evaluation horizons must contain at least two increasing positive values."
        )
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
    _validate_prediction_table(
        manifest,
        predictions,
        patient_column=patient_col,
        horizons=horizons,
    )
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
    cox_rows = []
    pairwise_rows = []
    roc_rows = []
    kaplan_meier_rows = []
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
            roc = _time_dependent_roc_table(
                training_outcome,
                outcome,
                model_frame["risk_score"].to_numpy(float),
                horizons,
            )
            roc.insert(0, "model", model_name)
            roc.insert(0, "cohort", cohort_name)
            roc_rows.extend(roc.to_dict(orient="records"))
            calibration_horizon = float(settings["calibration_horizon_months"])
            probability_column = f"pfs_{int(calibration_horizon)}m"
            calibration_input = model_frame.copy()
            calibration_input["progression_probability"] = (
                1.0 - calibration_input[probability_column]
            )
            calibration = _calibration_table(
                calibration_input,
                progression_probability_column="progression_probability",
                time_column=time_col,
                event_column=event_col,
                horizon=calibration_horizon,
                groups=int(settings["calibration_groups"]),
            )
            calibration.insert(0, "model", model_name)
            calibration.insert(0, "cohort", cohort_name)
            calibration_rows.extend(calibration.to_dict(orient="records"))
            risk_summary, coefficient_rows = _risk_stratification(
                model_frame,
                score_column="risk_score",
                cutoff=float(cutoffs[model_name]),
                time_column=time_col,
                event_column=event_col,
                adjustment_columns=adjustment_columns,
            )
            risk_rows.append(
                {"cohort": cohort_name, "model": model_name, **risk_summary}
            )
            kaplan_meier = _kaplan_meier_curve_table(
                model_frame,
                score_column="risk_score",
                cutoff=float(cutoffs[model_name]),
                time_column=time_col,
                event_column=event_col,
            )
            kaplan_meier.insert(0, "model", model_name)
            kaplan_meier.insert(0, "cohort", cohort_name)
            kaplan_meier_rows.extend(kaplan_meier.to_dict(orient="records"))
            for row in coefficient_rows:
                cox_rows.append({"cohort": cohort_name, "model": model_name, **row})
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
    risk_table = pd.DataFrame(risk_rows)
    pairwise_table = pd.DataFrame(pairwise_rows)
    if not risk_table.empty:
        risk_table["logrank_p_value_fdr"] = risk_table.groupby("cohort", sort=False)[
            "logrank_p_value"
        ].transform(_benjamini_hochberg)
        risk_table["adjusted_p_value_fdr"] = risk_table.groupby("cohort", sort=False)[
            "adjusted_p_value"
        ].transform(_benjamini_hochberg)
    if not pairwise_table.empty:
        pairwise_table["p_value_fdr"] = pairwise_table.groupby(
            ["cohort", "metric"], sort=False
        )["p_value"].transform(_benjamini_hochberg)
    tables = {
        "performance": pd.DataFrame(performance_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "risk_stratification": risk_table,
        "cox_regression": pd.DataFrame(cox_rows),
        "pairwise_comparisons": pairwise_table,
        "time_dependent_roc": pd.DataFrame(roc_rows),
        "kaplan_meier_curves": pd.DataFrame(kaplan_meier_rows),
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
                "multiplicity_adjustment": {
                    "method": "Benjamini-Hochberg",
                    "risk_stratification_family": "models within cohort",
                    "pairwise_comparison_family": "model pairs within cohort and metric",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tables
