from pathlib import Path

import numpy as np
import pandas as pd

from icvs_gbm_pfs.config import load_config
from icvs_gbm_pfs.descriptive import summarize_cohort_characteristics

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def test_cohort_summary_separates_nested_biological_subset() -> None:
    cohorts = np.repeat(["training", "temporal_validation", "spatial_validation"], 6)
    frame = pd.DataFrame(
        {
            "patient_id": [f"P{index:03d}" for index in range(18)],
            "cohort": cohorts,
            "pfs_months": np.tile([4.0, 7.0, 10.0, 13.0, 16.0, 19.0], 3),
            "pfs_event": np.tile([1, 1, 1, 0, 1, 0], 3),
            "biological_subset": [1, 1, 0, 0, 0, 0] + [0] * 12,
            "age_years": np.arange(50.0, 68.0),
            "sex": np.tile(["Female", "Male"], 9),
            "tumor_location": np.tile(["Frontal", "Temporal", "Parietal"], 6),
            "laterality": np.tile(["Left", "Right", "Midline/bilateral"], 6),
            "mgmt_methylated": np.tile([0, 1], 9),
            "non_gross_total_resection": np.tile([0, 1, 0], 6),
            "postoperative_treatment": np.tile(
                ["Stupp regimen", "Radiotherapy only", "Temozolomide only"], 6
            ),
        }
    )
    characteristics, comparisons = summarize_cohort_characteristics(
        frame, load_config(CONFIG_PATH)
    )
    assert set(characteristics["analysis_population"]) == {
        "primary",
        "nested_biological_subset",
    }
    assert "Follow-up, months" in set(characteristics["characteristic"])
    assert "Progression-free survival" in set(comparisons["characteristic"])
    assert comparisons["p_value"].dropna().between(0.0, 1.0).all()
    locations = characteristics.loc[
        characteristics["characteristic"].eq("Tumor location"), "level"
    ].drop_duplicates()
    assert locations.tolist() == [
        "Frontal",
        "Temporal",
        "Parietal",
        "Occipital",
        "Deep-seated",
    ]
