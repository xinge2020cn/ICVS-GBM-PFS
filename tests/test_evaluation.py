from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icvs_gbm_pfs.config import StudyConfig, load_config
from icvs_gbm_pfs.evaluation import _benjamini_hochberg, evaluate_models

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def evaluation_config() -> StudyConfig:
    original = load_config(CONFIG_PATH)
    values = deepcopy(original.values)
    values["evaluation"]["bootstrap_resamples"] = 10
    return StudyConfig(values=values, source=original.source)


def test_benjamini_hochberg_preserves_original_order() -> None:
    adjusted = _benjamini_hochberg(pd.Series([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])


def test_locked_evaluation_writes_all_primary_tables(tmp_path: Path) -> None:
    per_cohort = 60
    count = per_cohort * 3
    index = np.arange(count)
    within_cohort = index % per_cohort
    manifest = pd.DataFrame(
        {
            "patient_id": [f"P{value:03d}" for value in index],
            "cohort": np.repeat(
                ["training", "temporal_validation", "spatial_validation"], per_cohort
            ),
            "center_id": np.repeat(["C1", "C1", "C2"], per_cohort),
            "pfs_months": 1.0 + (within_cohort % 50),
            "pfs_event": (within_cohort % 4 != 0).astype(int),
            "biological_subset": np.zeros(count, dtype=int),
            "age_years": 45.0 + (within_cohort % 30),
            "mgmt_methylated": (within_cohort % 2).astype(int),
            "non_gross_total_resection": (within_cohort % 5 == 0).astype(int),
        }
    )
    risk = -(manifest["pfs_months"].to_numpy(float) / 12.0) + 0.1 * manifest["pfs_event"].to_numpy(
        float
    )
    predictions = pd.DataFrame(
        {
            "patient_id": manifest["patient_id"],
            "model": "locked_model",
            "risk_score": risk,
        }
    )
    for horizon in (6, 12, 18, 24, 30, 36):
        predictions[f"pfs_{horizon}m"] = np.exp(-0.02 * horizon * np.exp(risk - np.mean(risk)))
    tables = evaluate_models(
        manifest,
        predictions,
        evaluation_config(),
        tmp_path,
    )
    assert set(tables) == {
        "performance",
        "calibration",
        "risk_stratification",
        "cox_regression",
        "pairwise_comparisons",
    }
    assert len(tables["performance"]) == 3
    assert (tmp_path / "performance.csv").is_file()
    assert (tmp_path / "calibration.csv").is_file()
    assert (tmp_path / "cox_regression.csv").is_file()
    assert "logrank_p_value_fdr" in tables["risk_stratification"]
    assert {
        "hazard_ratio",
        "ci_low",
        "ci_high",
        "p_value",
        "proportional_hazards_p_value",
    }.issubset(tables["cox_regression"].columns)


def test_locked_evaluation_rejects_incomplete_patient_coverage(tmp_path: Path) -> None:
    per_cohort = 10
    count = per_cohort * 3
    index = np.arange(count)
    manifest = pd.DataFrame(
        {
            "patient_id": [f"P{value:03d}" for value in index],
            "cohort": np.repeat(
                ["training", "temporal_validation", "spatial_validation"], per_cohort
            ),
            "center_id": np.repeat(["C1", "C1", "C2"], per_cohort),
            "pfs_months": 2.0 + (index % per_cohort),
            "pfs_event": (index % 2).astype(int),
            "biological_subset": np.zeros(count, dtype=int),
            "age_years": np.full(count, 55.0),
            "mgmt_methylated": (index % 2).astype(int),
            "non_gross_total_resection": (index % 3 == 0).astype(int),
        }
    )
    predictions = pd.DataFrame(
        {
            "patient_id": manifest["patient_id"].iloc[:-1],
            "model": "locked_model",
            "risk_score": np.linspace(-1.0, 1.0, count - 1),
        }
    )
    for horizon in (6, 12, 18, 24, 30, 36):
        predictions[f"pfs_{horizon}m"] = np.exp(-0.01 * horizon)
    with pytest.raises(ValueError, match="exactly one prediction"):
        evaluate_models(manifest, predictions, evaluation_config(), tmp_path)
