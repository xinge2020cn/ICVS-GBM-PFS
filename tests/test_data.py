from pathlib import Path

import pandas as pd
import pytest

from icvs_gbm_pfs.config import load_config
from icvs_gbm_pfs.data import validate_manifest

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def valid_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "cohort": ["training", "temporal_validation", "spatial_validation"],
            "center_id": ["C1", "C1", "C2"],
            "pfs_months": [8.0, 10.0, 12.0],
            "pfs_event": [1, 0, 1],
            "biological_subset": [1, 0, 0],
        }
    )


def test_manifest_accepts_nested_subset_and_independent_spatial_center() -> None:
    audit = validate_manifest(valid_manifest(), load_config(CONFIG_PATH))
    assert audit.patients == 3
    assert audit.biological_subset == 1


def test_manifest_rejects_duplicate_patients() -> None:
    frame = valid_manifest()
    frame.loc[1, "patient_id"] = "P001"
    with pytest.raises(ValueError, match="not unique"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_rejects_spatial_center_overlap() -> None:
    frame = valid_manifest()
    frame.loc[2, "center_id"] = "C1"
    with pytest.raises(ValueError, match="overlap"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_rejects_biological_member_outside_training() -> None:
    frame = valid_manifest()
    frame.loc[1, "biological_subset"] = 1
    with pytest.raises(ValueError, match="nested"):
        validate_manifest(frame, load_config(CONFIG_PATH))
