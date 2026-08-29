from pathlib import Path

import numpy as np
import pandas as pd

from icvs_gbm_pfs.biological import prepare_biological_cohort
from icvs_gbm_pfs.config import load_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def test_biological_cohort_uses_training_oof_scores_and_locked_cutoff() -> None:
    manifest = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "cohort": [
                "training",
                "training",
                "training",
                "training",
                "temporal_validation",
                "spatial_validation",
            ],
            "biological_subset": [1, 1, 0, 0, 0, 0],
        }
    )
    scores = pd.DataFrame(
        {
            "patient_id": manifest["patient_id"],
            "vit_score_oof": [-2.0, 2.0, -1.0, 1.0, np.nan, np.nan],
        }
    )
    result = prepare_biological_cohort(manifest, scores, load_config(CONFIG_PATH))
    assert result["patient_id"].tolist() == ["P1", "P2"]
    assert result["vit_score"].tolist() == [-2.0, 2.0]
    assert result["vit_cutoff"].tolist() == [0.0, 0.0]
    assert result["vit_score_source"].eq("out_of_fold").all()
