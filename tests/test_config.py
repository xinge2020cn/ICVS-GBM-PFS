from pathlib import Path

import pytest

from icvs_gbm_pfs.config import load_config

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def test_study_config_contains_distinct_manifest_mappings() -> None:
    config = load_config(CONFIG_PATH)
    assert config.cohort("training") != config.cohort("spatial_validation")
    assert config.column("patient_id") != config.column("center_id")


def test_config_rejects_duplicate_cohort_labels(tmp_path: Path) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("spatial_validation: spatial_validation", "spatial_validation: training")
    path = tmp_path / "study.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="must be distinct"):
        load_config(path)
