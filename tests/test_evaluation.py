from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from icvs_gbm_pfs.config import StudyConfig, load_config
from icvs_gbm_pfs.evaluation import evaluate_models

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def evaluation_config() -> StudyConfig:
    original = load_config(CONFIG_PATH)
    values = deepcopy(original.values)
    values["evaluation"]["bootstrap_resamples"] = 10
    return StudyConfig(values=values, source=original.source)


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
        "pairwise_comparisons",
    }
    assert len(tables["performance"]) == 3
    assert (tmp_path / "performance.csv").is_file()
    assert (tmp_path / "calibration.csv").is_file()
