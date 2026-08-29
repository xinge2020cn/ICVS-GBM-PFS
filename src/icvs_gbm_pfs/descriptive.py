"""Cohort characteristics and survival follow-up summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times, qth_survival_time
from scipy import stats

from .config import StudyConfig


def _format_count(count: int, denominator: int) -> str:
    percentage = 100.0 * count / denominator if denominator else float("nan")
    return f"{count} ({percentage:.1f}%)"


def _format_continuous(values: pd.Series) -> str:
    numeric = values.to_numpy(dtype=float)
    return f"{np.mean(numeric):.1f} ({np.std(numeric, ddof=1):.1f})"


def _km_median_with_interval(
    durations: pd.Series,
    events: pd.Series,
) -> tuple[float, float, float]:
    estimator = KaplanMeierFitter(alpha=0.05).fit(
        durations.to_numpy(dtype=float),
        event_observed=events.to_numpy(dtype=bool),
    )
    interval = median_survival_times(estimator.confidence_interval_)
    limits = interval.to_numpy(dtype=float).reshape(-1)
    return float(estimator.median_survival_time_), float(limits[0]), float(limits[-1])


def _reverse_km_follow_up(
    durations: pd.Series,
    events: pd.Series,
) -> tuple[float, float, float]:
    estimator = KaplanMeierFitter().fit(
        durations.to_numpy(dtype=float),
        event_observed=~events.to_numpy(dtype=bool),
    )
    survival = estimator.survival_function_
    return (
        float(qth_survival_time(0.50, survival)),
        float(qth_survival_time(0.75, survival)),
        float(qth_survival_time(0.25, survival)),
    )


def _categorical_test(frame: pd.DataFrame, cohort_col: str, value_col: str) -> tuple[str, float]:
    contingency = pd.crosstab(frame[value_col], frame[cohort_col], dropna=False)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return "not_estimable", float("nan")
    chi_square, p_value, _, expected = stats.chi2_contingency(contingency, correction=False)
    if contingency.shape == (2, 2) and np.any(expected < 5):
        _, p_value = stats.fisher_exact(contingency.to_numpy())
        return "fisher_exact", float(p_value)
    return "pearson_chi_square", float(p_value)


def summarize_cohort_characteristics(
    frame: pd.DataFrame,
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create Table 1 source data and prespecified cross-cohort comparisons."""

    cohort_col = config.column("cohort")
    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    patient_col = config.column("patient_id")
    primary_cohorts = [
        config.cohort("training"),
        config.cohort("temporal_validation"),
        config.cohort("spatial_validation"),
    ]
    required = [
        patient_col,
        cohort_col,
        time_col,
        event_col,
        config.column("biological_subset"),
        config.column("age"),
        config.column("sex"),
        config.column("tumor_location"),
        config.column("laterality"),
        config.column("mgmt"),
        config.column("extent_of_resection"),
        config.column("postoperative_treatment"),
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Manifest is missing cohort-summary columns: {', '.join(missing)}")
    primary = frame.loc[frame[cohort_col].isin(primary_cohorts)].copy()
    if primary.empty or set(primary[cohort_col]) != set(primary_cohorts):
        raise ValueError("All three primary cohorts are required for cohort comparison.")
    if primary[patient_col].duplicated().any():
        raise ValueError("Cohort-summary patient identifiers must be unique.")

    populations = [("primary", primary)]
    biological = primary.loc[primary[config.column("biological_subset")].eq(1)].copy()
    if not biological.empty:
        populations.append(("nested_biological_subset", biological))

    rows: list[dict[str, object]] = []
    age_col = config.column("age")
    categorical = [
        (
            "Sex",
            config.column("sex"),
            {"Female": "Female", "Male": "Male"},
        ),
        (
            "Tumor location",
            config.column("tumor_location"),
            {
                "Frontal": "Frontal",
                "Temporal": "Temporal",
                "Parietal": "Parietal",
                "Occipital": "Occipital",
                "Deep-seated": "Deep-seated",
            },
        ),
        (
            "Laterality",
            config.column("laterality"),
            {
                "Left": "Left",
                "Right": "Right",
                "Midline/bilateral": "Midline/bilateral",
            },
        ),
        (
            "MGMT promoter methylation",
            config.column("mgmt"),
            {0: "Unmethylated", 1: "Methylated"},
        ),
        (
            "Extent of resection",
            config.column("extent_of_resection"),
            {0: "Gross total resection", 1: "Non-gross total resection"},
        ),
        (
            "Postoperative treatment",
            config.column("postoperative_treatment"),
            {
                "Stupp regimen": "Stupp regimen",
                "Radiotherapy only": "Radiotherapy only",
                "Temozolomide only": "Temozolomide only",
                "Other/none": "Other/none",
            },
        ),
    ]
    for population_name, population in populations:
        cohort_order = primary_cohorts if population_name == "primary" else ["biological_subset"]
        grouped_populations = (
            [("overall", population)]
            + [
                (cohort, population.loc[population[cohort_col].eq(cohort)])
                for cohort in cohort_order
            ]
            if population_name == "primary"
            else [("biological_subset", population)]
        )
        for cohort, group in grouped_populations:
            denominator = len(group)
            rows.append(
                {
                    "analysis_population": population_name,
                    "cohort": cohort,
                    "characteristic": "Patients",
                    "level": "All",
                    "value": str(denominator),
                    "n": denominator,
                }
            )
            rows.append(
                {
                    "analysis_population": population_name,
                    "cohort": cohort,
                    "characteristic": "Age, years",
                    "level": "Mean (SD)",
                    "value": _format_continuous(group[age_col]),
                    "n": denominator,
                }
            )
            for label, column, labels in categorical:
                levels = list(labels) if labels is not None else list(pd.unique(primary[column]))
                for level in levels:
                    count = int(group[column].eq(level).sum())
                    rows.append(
                        {
                            "analysis_population": population_name,
                            "cohort": cohort,
                            "characteristic": label,
                            "level": labels[level] if labels is not None else str(level),
                            "value": _format_count(count, denominator),
                            "n": denominator,
                        }
                    )
            event_count = int(group[event_col].sum())
            rows.append(
                {
                    "analysis_population": population_name,
                    "cohort": cohort,
                    "characteristic": "Progression-free survival event",
                    "level": "Event",
                    "value": _format_count(event_count, denominator),
                    "n": denominator,
                }
            )
            median_pfs, pfs_low, pfs_high = _km_median_with_interval(
                group[time_col], group[event_col]
            )
            rows.append(
                {
                    "analysis_population": population_name,
                    "cohort": cohort,
                    "characteristic": "Progression-free survival, months",
                    "level": "Kaplan-Meier median (95% CI)",
                    "value": f"{median_pfs:.1f} ({pfs_low:.1f}-{pfs_high:.1f})",
                    "n": denominator,
                }
            )
            follow_up, follow_up_q1, follow_up_q3 = _reverse_km_follow_up(
                group[time_col], group[event_col]
            )
            rows.append(
                {
                    "analysis_population": population_name,
                    "cohort": cohort,
                    "characteristic": "Follow-up, months",
                    "level": "Reverse Kaplan-Meier median (IQR)",
                    "value": f"{follow_up:.1f} ({follow_up_q1:.1f}-{follow_up_q3:.1f})",
                    "n": denominator,
                }
            )

    comparison_rows: list[dict[str, object]] = []
    age_groups = [
        primary.loc[primary[cohort_col].eq(cohort), age_col].to_numpy(dtype=float)
        for cohort in primary_cohorts
    ]
    _, age_p_value = stats.f_oneway(*age_groups)
    comparison_rows.append(
        {"characteristic": "Age, years", "test": "one_way_anova", "p_value": age_p_value}
    )
    for label, column, _ in categorical:
        test, p_value = _categorical_test(primary, cohort_col, column)
        comparison_rows.append(
            {"characteristic": label, "test": test, "p_value": p_value}
        )
    survival_comparison = multivariate_logrank_test(
        primary[time_col],
        primary[cohort_col],
        primary[event_col],
    )
    comparison_rows.append(
        {
            "characteristic": "Progression-free survival",
            "test": "multivariate_logrank",
            "p_value": float(survival_comparison.p_value),
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(comparison_rows)
