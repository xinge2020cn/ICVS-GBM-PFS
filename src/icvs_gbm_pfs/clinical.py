"""Clinical predictor selection and locked Cox comparator fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .config import StudyConfig
from .survival import structured_survival


@dataclass(frozen=True)
class _ClinicalFactor:
    key: str
    label: str
    columns: tuple[str, ...]
    level_labels: tuple[str, ...]
    reference: str | None = None


def _clinical_selection_design(
    frame: pd.DataFrame,
    config: StudyConfig,
) -> tuple[pd.DataFrame, tuple[_ClinicalFactor, ...]]:
    """Build the reference-coded design used for the reported clinical screen."""

    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    required = [
        time_col,
        event_col,
        config.column("age"),
        config.column("sex"),
        config.column("tumor_location"),
        config.column("laterality"),
        config.column("extent_of_resection"),
        config.column("mgmt"),
        config.column("postoperative_treatment"),
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Clinical selection is missing columns: {', '.join(missing)}")
    design = frame[required].copy()
    design[time_col] = pd.to_numeric(design[time_col], errors="raise")
    design[event_col] = pd.to_numeric(design[event_col], errors="raise")
    design["age_per_10_years"] = (
        pd.to_numeric(design[config.column("age")], errors="raise") - 60.0
    ) / 10.0
    design["male"] = design[config.column("sex")].eq("Male").astype(int)
    design["non_gtr"] = pd.to_numeric(
        design[config.column("extent_of_resection")], errors="raise"
    ).astype(int)
    design["mgmt_methylated"] = pd.to_numeric(
        design[config.column("mgmt")], errors="raise"
    ).astype(int)
    location = design[config.column("tumor_location")]
    design["loc_temporal"] = location.eq("Temporal").astype(int)
    design["loc_parietal"] = location.eq("Parietal").astype(int)
    design["loc_occipital"] = location.eq("Occipital").astype(int)
    design["loc_deep"] = location.eq("Deep-seated").astype(int)
    laterality = design[config.column("laterality")]
    design["lat_right"] = laterality.eq("Right").astype(int)
    design["lat_midline"] = laterality.eq("Midline/bilateral").astype(int)
    treatment = design[config.column("postoperative_treatment")]
    design["tx_rt"] = treatment.eq("Radiotherapy only").astype(int)
    design["tx_tmz"] = treatment.eq("Temozolomide only").astype(int)
    design["tx_other"] = treatment.eq("Other/none").astype(int)
    factors = (
        _ClinicalFactor(
            "age",
            "Age",
            ("age_per_10_years",),
            ("Age (per 10-year increase)",),
        ),
        _ClinicalFactor("sex", "Sex", ("male",), ("Male vs female",), "Female"),
        _ClinicalFactor(
            "tumor_location",
            "Tumor location",
            ("loc_temporal", "loc_parietal", "loc_occipital", "loc_deep"),
            ("Temporal", "Parietal", "Occipital", "Deep-seated"),
            "Frontal",
        ),
        _ClinicalFactor(
            "laterality",
            "Laterality",
            ("lat_right", "lat_midline"),
            ("Right", "Midline/bilateral"),
            "Left",
        ),
        _ClinicalFactor(
            "extent_of_resection",
            "Extent of resection",
            ("non_gtr",),
            ("Non-GTR vs GTR",),
            "GTR",
        ),
        _ClinicalFactor(
            "mgmt",
            "MGMT promoter methylation",
            ("mgmt_methylated",),
            ("Methylated vs unmethylated",),
            "Unmethylated",
        ),
        _ClinicalFactor(
            "postoperative_treatment",
            "Postoperative treatment",
            ("tx_rt", "tx_tmz", "tx_other"),
            ("Radiotherapy only", "Temozolomide only", "Other/none"),
            "Stupp regimen",
        ),
    )
    numeric = design[[time_col, event_col, *(c for factor in factors for c in factor.columns)]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("Clinical selection contains missing or nonfinite values.")
    return design, factors


def _fit_selection_model(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    time_column: str,
    event_column: str,
    penalizer: float,
) -> CoxPHFitter:
    model = CoxPHFitter(penalizer=penalizer)
    return model.fit(
        frame[[time_column, event_column, *columns]],
        duration_col=time_column,
        event_col=event_column,
    )


def _run_clinical_selection(
    training: pd.DataFrame,
    config: StudyConfig,
) -> tuple[pd.DataFrame, list[str]]:
    design, factors = _clinical_selection_design(training, config)
    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    settings = config.section("clinical")
    penalizer = float(settings["selection_penalizer"])
    univariable_threshold = float(settings["univariable_p_threshold"])
    multivariable_threshold = float(settings["multivariable_p_threshold"])
    univariable: dict[str, dict[str, float]] = {}
    group_p_values: dict[str, float] = {}
    candidates: list[str] = []
    for factor in factors:
        model = _fit_selection_model(
            design,
            factor.columns,
            time_column=time_col,
            event_column=event_col,
            penalizer=penalizer,
        )
        if len(factor.columns) == 1:
            group_p = float(model.summary.loc[factor.columns[0], "p"])
        else:
            group_p = float(model.log_likelihood_ratio_test().p_value)
        group_p_values[factor.key] = group_p
        for column in factor.columns:
            summary = model.summary.loc[column]
            univariable[column] = {
                "hazard_ratio": float(summary["exp(coef)"]),
                "ci_low": float(summary["exp(coef) lower 95%"]),
                "ci_high": float(summary["exp(coef) upper 95%"]),
                "p_value": float(summary["p"]),
            }
        if group_p < univariable_threshold:
            candidates.extend(factor.columns)
    if not candidates:
        raise RuntimeError("No clinical factor passed the prespecified univariable threshold.")
    multivariable = _fit_selection_model(
        design,
        candidates,
        time_column=time_col,
        event_column=event_col,
        penalizer=penalizer,
    )
    multivariable_group_p: dict[str, float] = {}
    for factor in factors:
        if not set(factor.columns).issubset(candidates):
            multivariable_group_p[factor.key] = float("nan")
        elif len(factor.columns) == 1:
            multivariable_group_p[factor.key] = float(
                multivariable.summary.loc[factor.columns[0], "p"]
            )
        else:
            reduced_columns = [column for column in candidates if column not in factor.columns]
            if reduced_columns:
                reduced = _fit_selection_model(
                    design,
                    reduced_columns,
                    time_column=time_col,
                    event_column=event_col,
                    penalizer=penalizer,
                )
                statistic = max(
                    0.0,
                    2.0 * (multivariable.log_likelihood_ - reduced.log_likelihood_),
                )
            else:
                statistic = float(multivariable.log_likelihood_ratio_test().test_statistic)
            multivariable_group_p[factor.key] = float(
                stats.chi2.sf(statistic, len(factor.columns))
            )
    retained = [
        factor.key
        for factor in factors
        if set(factor.columns).issubset(candidates)
        and multivariable_group_p[factor.key] < multivariable_threshold
    ]
    rows: list[dict[str, object]] = []
    for factor in factors:
        entered = set(factor.columns).issubset(candidates)
        retained_factor = factor.key in retained
        if len(factor.columns) > 1:
            rows.append(
                {
                    "row_type": "group",
                    "variable": factor.label,
                    "univariable_hr": np.nan,
                    "univariable_ci_low": np.nan,
                    "univariable_ci_high": np.nan,
                    "univariable_p": group_p_values[factor.key],
                    "entered_multivariable": entered,
                    "multivariable_hr": np.nan,
                    "multivariable_ci_low": np.nan,
                    "multivariable_ci_high": np.nan,
                    "multivariable_p": multivariable_group_p[factor.key],
                    "retained_clinical_model": retained_factor,
                }
            )
            rows.append(
                {
                    "row_type": "reference",
                    "variable": f"   {factor.reference} (reference)",
                    "univariable_hr": 1.0,
                    "univariable_ci_low": np.nan,
                    "univariable_ci_high": np.nan,
                    "univariable_p": np.nan,
                    "entered_multivariable": entered,
                    "multivariable_hr": 1.0 if entered else np.nan,
                    "multivariable_ci_low": np.nan,
                    "multivariable_ci_high": np.nan,
                    "multivariable_p": np.nan,
                    "retained_clinical_model": False,
                }
            )
        for column, level_label in zip(
            factor.columns, factor.level_labels, strict=True
        ):
            summary = multivariable.summary.loc[column] if column in candidates else None
            variable = level_label
            if len(factor.columns) > 1:
                variable = f"   {level_label} vs {str(factor.reference).lower()}"
            rows.append(
                {
                    "row_type": "level",
                    "variable": variable,
                    "univariable_hr": univariable[column]["hazard_ratio"],
                    "univariable_ci_low": univariable[column]["ci_low"],
                    "univariable_ci_high": univariable[column]["ci_high"],
                    "univariable_p": univariable[column]["p_value"],
                    "entered_multivariable": column in candidates,
                    "multivariable_hr": (
                        float(summary["exp(coef)"]) if summary is not None else np.nan
                    ),
                    "multivariable_ci_low": (
                        float(summary["exp(coef) lower 95%"])
                        if summary is not None
                        else np.nan
                    ),
                    "multivariable_ci_high": (
                        float(summary["exp(coef) upper 95%"])
                        if summary is not None
                        else np.nan
                    ),
                    "multivariable_p": (
                        float(summary["p"]) if summary is not None else np.nan
                    ),
                    "retained_clinical_model": (
                        retained_factor and len(factor.columns) == 1
                    ),
                }
            )
    return pd.DataFrame(rows), retained


def clinical_selection_table(
    manifest: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    """Return the complete training-cohort clinical screening table."""

    cohort_col = config.column("cohort")
    training = manifest.loc[
        manifest[cohort_col].eq(config.cohort("training"))
    ].reset_index(drop=True)
    if training.empty:
        raise ValueError("The training cohort is empty.")
    table, _ = _run_clinical_selection(training, config)
    return table


def _clinical_matrix(frame: pd.DataFrame, config: StudyConfig) -> np.ndarray:
    age_per_decade = pd.to_numeric(
        frame[config.column("age")], errors="raise"
    ).to_numpy(float) / 10.0
    mgmt = pd.to_numeric(frame[config.column("mgmt")], errors="raise").to_numpy(float)
    non_gtr = pd.to_numeric(
        frame[config.column("extent_of_resection")], errors="raise"
    ).to_numpy(float)
    matrix = np.column_stack([age_per_decade, mgmt, non_gtr])
    if not np.isfinite(matrix).all():
        raise ValueError("Clinical predictors contain missing or nonfinite values.")
    if not set(np.unique(mgmt)).issubset({0.0, 1.0}):
        raise ValueError("MGMT promoter methylation must use binary zero-one coding.")
    if not set(np.unique(non_gtr)).issubset({0.0, 1.0}):
        raise ValueError("Extent of resection must use binary zero-one coding.")
    if np.any(age_per_decade < 1.8):
        raise ValueError("Age values must be at least 18 years.")
    return matrix


def fit_clinical_model(
    manifest: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Reproduce clinical screening and fit the locked retained predictors."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    training_mask = manifest[cohort_col].eq(config.cohort("training")).to_numpy()
    training = manifest.loc[training_mask]
    if training.empty:
        raise ValueError("The training cohort is empty.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selection, retained = _run_clinical_selection(training.reset_index(drop=True), config)
    selection.to_csv(output / "clinical_selection.csv", index=False)
    clinical_settings = config.section("clinical")
    expected = list(clinical_settings["expected_retained_predictors"])
    selection_matches_expected = set(retained) == set(expected)
    if (
        bool(clinical_settings["enforce_expected_retained_predictors"])
        and not selection_matches_expected
    ):
        raise RuntimeError(
            "Clinical selection did not reproduce the locked retained predictors. "
            f"Expected {expected}; observed {retained}."
        )
    training_features = _clinical_matrix(training, config)
    all_features = _clinical_matrix(manifest, config)
    outcome = structured_survival(
        training[config.column("pfs_event")], training[config.column("pfs_time")]
    )
    model = CoxPHSurvivalAnalysis(alpha=1e-8, ties="breslow").fit(training_features, outcome)
    risk = model.predict(all_features)
    horizons = np.asarray(config.section("icvs")["horizons_months"], dtype=float)
    if (
        horizons.ndim != 1
        or horizons.size == 0
        or not np.isfinite(horizons).all()
        or np.any(horizons <= 0)
        or np.any(np.diff(horizons) <= 0)
    ):
        raise ValueError("Clinical-model horizons must be strictly increasing positive values.")
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
            "clinical_selection": selection,
            "retained_predictors": retained,
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
                "univariable_p_threshold": float(
                    clinical_settings["univariable_p_threshold"]
                ),
                "multivariable_p_threshold": float(
                    clinical_settings["multivariable_p_threshold"]
                ),
                "selection_penalizer": float(clinical_settings["selection_penalizer"]),
                "expected_retained_predictors": expected,
                "observed_retained_predictors": retained,
                "selection_matches_expected": selection_matches_expected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
