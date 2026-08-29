from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from icvs_gbm_pfs.config import load_config
from icvs_gbm_pfs.segmentation import (
    assemble_segmentation_manifests,
    collect_nnunet_oof_predictions,
    prepare_nnunet_inference,
    segmentation_bland_altman,
    segmentation_metrics,
)

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


def test_volume_agreement_reports_bias_and_limits_by_cohort() -> None:
    patient_metrics = pd.DataFrame(
        {
            "cohort": ["training", "training"],
            "reference_volume_ml": [10.0, 20.0],
            "prediction_volume_ml": [12.0, 18.0],
            "signed_relative_volume_error": [0.20, -0.10],
        }
    )
    result = segmentation_bland_altman(patient_metrics, "cohort")
    absolute = result.loc[result["scale"].eq("absolute_volume")].iloc[0]
    assert absolute["bias"] == pytest.approx(0.0)
    assert absolute["standard_deviation"] == pytest.approx(np.sqrt(8.0))
    assert set(result["scale"]) == {"absolute_volume", "relative_volume"}


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


def test_collects_one_out_of_fold_prediction_per_training_patient(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    dataset_name = "Dataset501_GBMTumorCore"
    raw_dataset = tmp_path / "raw" / dataset_name
    raw_dataset.mkdir(parents=True)
    pd.DataFrame(
        {
            "case_id": ["GBMTC_0001", "GBMTC_0002"],
            "patient_id": ["P001", "P002"],
        }
    ).to_csv(raw_dataset / "case_mapping.csv", index=False)
    model_dir = (
        tmp_path
        / "results"
        / dataset_name
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    prediction_one = model_dir / "fold_0" / "validation" / "GBMTC_0001.nii.gz"
    prediction_two = model_dir / "fold_1" / "validation" / "GBMTC_0002.nii.gz"
    prediction_one.parent.mkdir(parents=True)
    prediction_two.parent.mkdir(parents=True)
    prediction_one.write_bytes(b"prediction-one")
    prediction_two.write_bytes(b"prediction-two")
    frame = pd.DataFrame(
        {
            "patient_id": ["P001", "P002"],
            "cohort": ["training", "training"],
            "preprocessed_tumor_mask_path": ["reference-one.nii.gz", "reference-two.nii.gz"],
        }
    )
    output = tmp_path / "training_segmentation.csv"

    result = collect_nnunet_oof_predictions(
        frame,
        config,
        nnunet_raw=tmp_path / "raw",
        nnunet_results=tmp_path / "results",
        output_manifest=output,
    )

    assert set(result["patient_id"]) == {"P001", "P002"}
    assert set(result["cohort"]) == {"training"}
    assert result["prediction_mask_path"].str.contains("fold_").all()
    assert output.is_file()


def test_assembles_training_and_locked_validation_segmentation_manifests(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training.csv"
    validation = tmp_path / "validation.csv"
    columns = [
        "case_id",
        "patient_id",
        "cohort",
        "reference_mask_path",
        "prediction_mask_path",
    ]
    pd.DataFrame(
        [["T001", "P001", "training", "r1.nii.gz", "p1.nii.gz"]],
        columns=columns,
    ).to_csv(training, index=False)
    pd.DataFrame(
        [
            ["V001", "P002", "temporal_validation", "r2.nii.gz", "p2.nii.gz"],
            ["V002", "P003", "spatial_validation", "r3.nii.gz", "p3.nii.gz"],
        ],
        columns=columns,
    ).to_csv(validation, index=False)
    output = tmp_path / "assembled.csv"

    result = assemble_segmentation_manifests(
        training,
        validation,
        load_config(CONFIG_PATH),
        output,
    )

    assert len(result) == 3
    assert result["patient_id"].is_unique
    assert output.is_file()
