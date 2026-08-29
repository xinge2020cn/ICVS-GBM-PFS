from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from icvs_gbm_pfs.config import load_config
from icvs_gbm_pfs.segmentation import prepare_nnunet_inference, segmentation_metrics

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "study.yaml"


def test_identical_masks_have_perfect_metrics() -> None:
    mask = np.zeros((8, 16, 16), dtype=bool)
    mask[2:6, 4:12, 4:12] = True
    metrics = segmentation_metrics(mask, mask, (5.0, 1.0, 1.0))
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["surface_dice"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["hd95_mm"] == pytest.approx(0.0)
    assert metrics["signed_relative_volume_error"] == pytest.approx(0.0)


def test_missing_prediction_is_reported_as_complete_failure() -> None:
    reference = np.zeros((8, 16, 16), dtype=bool)
    reference[2:6, 4:12, 4:12] = True
    prediction = np.zeros_like(reference)
    metrics = segmentation_metrics(reference, prediction, (5.0, 1.0, 1.0))
    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["surface_dice"] == pytest.approx(0.0)
    assert metrics["sensitivity"] == pytest.approx(0.0)
    assert np.isinf(metrics["hd95_mm"])


def test_segmentation_metrics_reject_invalid_spacing() -> None:
    mask = np.ones((2, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="spacing"):
        segmentation_metrics(mask, mask, (1.0, 0.0, 1.0))


def test_prepare_nnunet_inference_writes_validation_mapping(tmp_path: Path) -> None:
    image = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.float32))
    mask = sitk.GetImageFromArray(np.ones((2, 3, 4), dtype=np.uint8))
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(mask, str(mask_path))
    frame = pd.DataFrame(
        {
            "patient_id": ["T001", "S001"],
            "cohort": ["temporal_validation", "spatial_validation"],
            "preprocessed_t1_path": [image_path, image_path],
            "preprocessed_t2_path": [image_path, image_path],
            "preprocessed_flair_path": [image_path, image_path],
            "preprocessed_ce_t1_path": [image_path, image_path],
            "preprocessed_tumor_mask_path": [mask_path, mask_path],
        }
    )
    input_dir = tmp_path / "input"
    prediction_dir = tmp_path / "predictions"
    manifest_path = tmp_path / "segmentation_manifest.csv"

    mapping = prepare_nnunet_inference(
        frame,
        load_config(CONFIG_PATH),
        input_dir=input_dir,
        prediction_dir=prediction_dir,
        output_manifest=manifest_path,
    )

    assert len(list(input_dir.glob("*.nii.gz"))) == 8
    assert set(mapping["case_id"]) == {"GBMIV_0001", "GBMIV_0002"}
    assert set(mapping["cohort"]) == {"temporal_validation", "spatial_validation"}
    assert manifest_path.is_file()
