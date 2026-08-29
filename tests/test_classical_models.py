from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from deployment.predictor import ICVSPredictor
from icvs_gbm_pfs.clinical import fit_clinical_model
from icvs_gbm_pfs.config import StudyConfig, load_config
from icvs_gbm_pfs.icvs import fit_icvs_model

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
            "sex": np.where((index // 2) % 2 == 0, "Female", "Male"),
            "tumor_location": np.array(
                ["Frontal", "Temporal", "Parietal", "Occipital", "Deep-seated"]
            )[(index // 3) % 5],
            "laterality": np.array(["Left", "Right", "Midline/bilateral"])[
                (index // 4) % 3
            ],
            "mgmt_methylated": (index % 2).astype(int),
            "non_gross_total_resection": (index % 7 == 0).astype(int),
            "postoperative_treatment": np.array(
                [
                    "Stupp regimen",
                    "Radiotherapy only",
                    "Temozolomide only",
                    "Other/none",
                ]
            )[(index // 5) % 4],
        }
    )


def reduced_tree_config() -> StudyConfig:
    original = load_config(CONFIG_PATH)
    values = deepcopy(original.values)
    values["icvs"]["n_estimators"] = 25
    return StudyConfig(values=values, source=original.source)


def clinical_test_config() -> StudyConfig:
    original = load_config(CONFIG_PATH)
    values = deepcopy(original.values)
    values["clinical"]["univariable_p_threshold"] = 1.0
    values["clinical"]["multivariable_p_threshold"] = 1.0
    values["clinical"]["enforce_expected_retained_predictors"] = False
    return StudyConfig(values=values, source=original.source)


def test_clinical_model_writes_bounded_survival_probabilities(tmp_path: Path) -> None:
    predictions = fit_clinical_model(model_frame(), clinical_test_config(), tmp_path)
    probability_columns = [column for column in predictions if "_pfs_" in column]
    values = predictions[probability_columns].to_numpy(float)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert (tmp_path / "clinical_model.joblib").is_file()
    selection = pd.read_csv(tmp_path / "clinical_selection.csv")
    assert set(selection["row_type"]) == {"group", "reference", "level"}
    assert "Postoperative treatment" in set(selection["variable"].str.strip())


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
    deployed = ICVSPredictor(tmp_path / "icvs_model.joblib").predict(
        {
            "age_years": 60.0,
            "mgmt_promoter_methylated": True,
            "extent_of_resection": "gross_total",
            "vit_score_standardized": 0.0,
        }
    )
    assert 0.0 <= deployed["pfs_probability_12m"] <= 1.0
