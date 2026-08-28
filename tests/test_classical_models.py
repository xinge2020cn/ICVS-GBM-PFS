from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from dlr3.clinical import fit_clinical_model
from dlr3.config import StudyConfig, load_config
from dlr3.icvs import fit_icvs_model

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def model_frame() -> pd.DataFrame:
    count = 90
    index = np.arange(count)
    cohort = np.repeat(["training", "temporal_validation", "spatial_validation"], [50, 20, 20])
    center = np.repeat(["C1", "C1", "C2"], [50, 20, 20])
    return pd.DataFrame(
        {
            "patient_id": [f"P{value:03d}" for value in index],
            "cohort": cohort,
            "center_id": center,
            "pfs_months": 6.0 + (index % 45),
            "pfs_event": (index % 3 != 0).astype(int),
            "biological_subset": np.zeros(count, dtype=int),
            "age_years": 40.0 + (index % 35),
            "mgmt_methylated": (index % 2).astype(int),
            "non_gross_total_resection": (index % 4 == 0).astype(int),
        }
    )


def reduced_tree_config() -> StudyConfig:
    original = load_config(CONFIG_PATH)
    values = deepcopy(original.values)
    values["icvs"]["n_estimators"] = 25
    return StudyConfig(values=values, source=original.source)


def test_clinical_model_writes_bounded_survival_probabilities(tmp_path: Path) -> None:
    predictions = fit_clinical_model(model_frame(), load_config(CONFIG_PATH), tmp_path)
    probability_columns = [column for column in predictions if "_pfs_" in column]
    values = predictions[probability_columns].to_numpy(float)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert (tmp_path / "clinical_model.joblib").is_file()


def test_icvs_uses_oob_training_predictions_and_locked_validation(tmp_path: Path) -> None:
    frame = model_frame()
    index = np.arange(len(frame), dtype=float)
    scores = pd.DataFrame(
        {
            "patient_id": frame["patient_id"],
            "vit_score_oof": np.where(index < 50, np.sin(index / 7.0), np.nan),
            "vit_score_final": np.cos(index / 9.0) + index / 100.0,
        }
    )
    predictions = fit_icvs_model(frame, scores, reduced_tree_config(), tmp_path)
    probability_columns = [column for column in predictions if "_pfs_" in column]
    values = predictions[probability_columns].to_numpy(float)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert predictions.loc[:49, "icvs_risk_score"].notna().all()
    assert (tmp_path / "icvs_model.joblib").is_file()
