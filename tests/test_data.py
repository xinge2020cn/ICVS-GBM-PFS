from pathlib import Path

import pandas as pd
import pytest

from icvs_gbm_pfs.config import load_config
from icvs_gbm_pfs.data import read_manifest, validate_manifest

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
            "age_years": [54.0, 61.0, 48.0],
            "mgmt_methylated": [1, 0, 1],
            "non_gross_total_resection": [0, 1, 0],
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


@pytest.mark.parametrize("value", [0.5, 1.5, -1])
def test_manifest_rejects_nonbinary_event_values(value: float) -> None:
    frame = valid_manifest()
    frame["pfs_event"] = frame["pfs_event"].astype(float)
    frame.loc[0, "pfs_event"] = value
    with pytest.raises(ValueError, match="event values must be binary"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_rejects_nonfinite_survival_time() -> None:
    frame = valid_manifest()
    frame.loc[0, "pfs_months"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_rejects_case_insensitive_patient_collision() -> None:
    frame = valid_manifest()
    frame.loc[1, "patient_id"] = "p001"
    with pytest.raises(ValueError, match="case-insensitive"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_rejects_reserved_filesystem_name() -> None:
    frame = valid_manifest()
    frame.loc[0, "patient_id"] = "CON"
    with pytest.raises(ValueError, match="reserved filesystem"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_rejects_invalid_clinical_coding() -> None:
    frame = valid_manifest()
    frame["mgmt_methylated"] = frame["mgmt_methylated"].astype(float)
    frame.loc[0, "mgmt_methylated"] = 0.5
    with pytest.raises(ValueError, match="MGMT"):
        validate_manifest(frame, load_config(CONFIG_PATH))


def test_manifest_reader_preserves_leading_zero_patient_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"patient_id": ["001", "002"], "value": [1, 2]}).to_csv(path, index=False)
    frame = read_manifest(path)
    assert frame["patient_id"].tolist() == ["001", "002"]
