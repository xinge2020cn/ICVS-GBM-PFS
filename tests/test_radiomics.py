from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sksurv.linear_model import CoxnetSurvivalAnalysis

from icvs_gbm_pfs.config import StudyConfig, load_config
from icvs_gbm_pfs.radiomics import (
    _radiomic_feature_name,
    _relative_penalty_path,
    fit_radiomics_model,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def test_shape_features_are_modality_independent() -> None:
    name = "original_shape_MeshVolume"
    assert _radiomic_feature_name("t1", name) == "shape__original_shape_MeshVolume"
    assert _radiomic_feature_name("ce_t1", name) == "shape__original_shape_MeshVolume"


def test_intensity_and_texture_features_retain_modality() -> None:
    assert (
        _radiomic_feature_name("flair", "wavelet-HLL_glcm_Contrast")
        == "flair__wavelet-HLL_glcm_Contrast"
    )


def test_radiomics_configuration_uses_isotropic_resampling() -> None:
    path = Path(__file__).parents[1] / "configs" / "radiomics.yaml"
    parameters = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert parameters["setting"]["resampledPixelSpacing"] == [1.0, 1.0, 1.0]


def test_relative_penalty_path_is_normalized_and_decreasing() -> None:
    model = CoxnetSurvivalAnalysis(alphas=[0.5, 0.25, 0.125])
    model.alphas_ = np.array([0.5, 0.25, 0.125])
    assert np.allclose(_relative_penalty_path(model), [1.0, 0.5, 0.25])


def test_radiomics_screening_is_recorded_for_every_cross_validation_fold(
    tmp_path: Path,
) -> None:
    count = 90
    index = np.arange(count)
    rng = np.random.default_rng(2026)
    signal = rng.normal(size=count)
    event_time = rng.exponential(scale=np.exp(-0.5 * signal) * 12.0) + 1.0
    manifest = pd.DataFrame(
        {
            "patient_id": [f"P{value:03d}" for value in index],
            "cohort": np.repeat(
                ["training", "temporal_validation", "spatial_validation"], [60, 15, 15]
            ),
            "center_id": np.repeat(["C1", "C1", "C2"], [60, 15, 15]),
            "pfs_months": event_time,
            "pfs_event": (rng.random(count) < 0.80).astype(int),
            "biological_subset": np.zeros(count, dtype=int),
            "age_years": np.full(count, 55.0),
            "mgmt_methylated": index % 2,
            "non_gross_total_resection": index % 3 == 0,
        }
    )
    features = pd.DataFrame(
        {
            "patient_id": manifest["patient_id"],
            "feature_signal": signal,
            "feature_noise_1": rng.normal(size=count),
            "feature_noise_2": rng.normal(size=count),
        }
    )
    original = load_config(CONFIG_PATH)
    values = deepcopy(original.values)
    values["radiomics"]["univariable_p_threshold"] = 1.0
    values["radiomics"]["cross_validation_folds"] = 3
    values["radiomics"]["use_one_standard_error_rule"] = False
    config = StudyConfig(values=values, source=original.source)

    scores = fit_radiomics_model(manifest, features, config, tmp_path)

    screening = pd.read_csv(tmp_path / "radiomics_fold_screening.csv")
    assignments = pd.read_csv(tmp_path / "radiomics_cross_validation_assignments.csv")
    assert set(screening["fold"]) == {0, 1, 2}
    assert screening.groupby("fold")["passed_threshold"].any().all()
    assert len(assignments) == 60
    assert assignments["patient_id"].is_unique
    assert scores["radiomics_score"].notna().all()
