"""nnU-Net data preparation, execution, and patient-level segmentation assessment."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from .config import StudyConfig


def prepare_nnunet_dataset(
    frame: pd.DataFrame,
    config: StudyConfig,
    nnunet_raw: str | Path,
) -> Path:
    """Create the nnU-Net raw training layout from the development cohort only."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    training = frame.loc[frame[cohort_col].eq(config.cohort("training"))].copy()
    if training.empty:
        raise ValueError("The training cohort is empty.")
    required = [
        patient_col,
        "preprocessed_t1_path",
        "preprocessed_t2_path",
        "preprocessed_flair_path",
        "preprocessed_ce_t1_path",
        "preprocessed_tumor_mask_path",
    ]
    missing = [column for column in required if column not in training]
    if missing:
        raise ValueError(f"Manifest is missing nnU-Net columns: {', '.join(missing)}")
    dataset_id = int(config.section("segmentation")["dataset_id"])
    dataset_dir = Path(nnunet_raw).resolve() / f"Dataset{dataset_id:03d}_GBMTumorCore"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for index, row in enumerate(training.to_dict(orient="records"), start=1):
        case_id = f"GBMTC_{index:04d}"
        for channel, column in enumerate(
            (
                "preprocessed_t1_path",
                "preprocessed_t2_path",
                "preprocessed_flair_path",
                "preprocessed_ce_t1_path",
            )
        ):
            shutil.copy2(row[column], images_dir / f"{case_id}_{channel:04d}.nii.gz")
        shutil.copy2(row["preprocessed_tumor_mask_path"], labels_dir / f"{case_id}.nii.gz")
        mapping.append({"case_id": case_id, patient_col: str(row[patient_col])})
    dataset_json = {
        "channel_names": {"0": "T1WI", "1": "T2WI", "2": "T2-FLAIR", "3": "CE-T1WI"},
        "labels": {"background": 0, "tumor_core": 1},
        "numTraining": len(training),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(mapping).to_csv(dataset_dir / "case_mapping.csv", index=False)
    return dataset_dir


@contextmanager
def nnunet_environment(
    raw: str | Path,
    preprocessed: str | Path,
    results: str | Path,
) -> Iterator[dict[str, str]]:
    """Provide explicit nnU-Net storage locations without changing global state."""

    environment = os.environ.copy()
    environment["nnUNet_raw"] = str(Path(raw).resolve())
    environment["nnUNet_preprocessed"] = str(Path(preprocessed).resolve())
    environment["nnUNet_results"] = str(Path(results).resolve())
    yield environment


def run_nnunet_training(
    config: StudyConfig,
    *,
    raw: str | Path,
    preprocessed: str | Path,
    results: str | Path,
    trainer: str = "nnUNetTrainer",
) -> None:
    """Plan, preprocess, and train the five locked nnU-Net folds."""

    settings = config.section("segmentation")
    dataset_id = str(int(settings["dataset_id"]))
    configuration = str(settings["configuration"])
    with nnunet_environment(raw, preprocessed, results) as environment:
        subprocess.run(
            ["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity"],
            check=True,
            env=environment,
        )
        _validate_nnunet_plan(config, preprocessed)
        for fold in settings["folds"]:
            subprocess.run(
                [
                    "nnUNetv2_train",
                    dataset_id,
                    configuration,
                    str(int(fold)),
                    "-tr",
                    trainer,
                ],
                check=True,
                env=environment,
            )


def _validate_nnunet_plan(config: StudyConfig, preprocessed: str | Path) -> None:
    """Block training when the generated plan differs from the reported patch and batch sizes."""

    settings = config.section("segmentation")
    dataset_id = int(settings["dataset_id"])
    dataset_dir = Path(preprocessed).resolve() / f"Dataset{dataset_id:03d}_GBMTumorCore"
    candidates = sorted(dataset_dir.glob("*Plans.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one nnU-Net plans file in {dataset_dir}, found {len(candidates)}."
        )
    plans = json.loads(candidates[0].read_text(encoding="utf-8"))
    configuration_name = str(settings["configuration"])
    try:
        generated = plans["configurations"][configuration_name]
    except KeyError as error:
        raise ValueError(f"nnU-Net plan has no '{configuration_name}' configuration.") from error
    expected_patch = [int(value) for value in settings["patch_shape_dhw"]]
    observed_patch = [int(value) for value in generated["patch_size"]]
    if observed_patch != expected_patch:
        raise ValueError(
            f"nnU-Net patch size is {observed_patch}; the locked study value is {expected_patch}."
        )
    expected_batch = int(settings["batch_size"])
    observed_batch = int(generated["batch_size"])
    if observed_batch != expected_batch:
        raise ValueError(
            f"nnU-Net batch size is {observed_batch}; the locked study value is {expected_batch}."
        )


def run_nnunet_prediction(
    config: StudyConfig,
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    raw: str | Path,
    preprocessed: str | Path,
    results: str | Path,
    trainer: str = "nnUNetTrainer",
) -> None:
    """Apply the locked five-fold ensemble to an inference directory."""

    settings = config.section("segmentation")
    command = [
        "nnUNetv2_predict",
        "-i",
        str(Path(input_dir).resolve()),
        "-o",
        str(Path(output_dir).resolve()),
        "-d",
        str(int(settings["dataset_id"])),
        "-c",
        str(settings["configuration"]),
        "-tr",
        trainer,
        "-f",
        *[str(int(fold)) for fold in settings["folds"]],
    ]
    with nnunet_environment(raw, preprocessed, results) as environment:
        subprocess.run(command, check=True, env=environment)


def segmentation_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    spacing_dhw: tuple[float, float, float],
    *,
    surface_tolerance_mm: float = 2.0,
) -> dict[str, float]:
    """Calculate overlap, boundary, detection, and volume metrics in three dimensions."""

    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction masks must have identical shapes.")
    if not reference.any():
        raise ValueError("Reference mask contains no foreground voxels.")
    intersection = np.logical_and(reference, prediction).sum(dtype=np.float64)
    reference_count = reference.sum(dtype=np.float64)
    prediction_count = prediction.sum(dtype=np.float64)
    dice = 2.0 * intersection / (reference_count + prediction_count)
    sensitivity = intersection / reference_count
    spacing = np.asarray(spacing_dhw, dtype=float)
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    reference_volume = reference_count * voxel_volume_ml
    prediction_volume = prediction_count * voxel_volume_ml
    signed_relative_error = (prediction_volume - reference_volume) / reference_volume
    if prediction.any():
        reference_surface = np.logical_xor(
            reference, ndimage.binary_erosion(reference, structure=np.ones((3, 3, 3)))
        )
        prediction_surface = np.logical_xor(
            prediction, ndimage.binary_erosion(prediction, structure=np.ones((3, 3, 3)))
        )
        distance_to_prediction = ndimage.distance_transform_edt(
            ~prediction_surface, sampling=spacing
        )
        distance_to_reference = ndimage.distance_transform_edt(~reference_surface, sampling=spacing)
        reference_distances = distance_to_prediction[reference_surface]
        prediction_distances = distance_to_reference[prediction_surface]
        all_distances = np.concatenate([reference_distances, prediction_distances])
        hd95 = float(np.percentile(all_distances, 95))
        surface_dice = float(
            (
                np.count_nonzero(reference_distances <= surface_tolerance_mm)
                + np.count_nonzero(prediction_distances <= surface_tolerance_mm)
            )
            / (reference_distances.size + prediction_distances.size)
        )
    else:
        hd95 = float("inf")
        surface_dice = 0.0
    return {
        "dice": float(dice),
        "surface_dice": surface_dice,
        "sensitivity": float(sensitivity),
        "hd95_mm": hd95,
        "reference_volume_ml": float(reference_volume),
        "prediction_volume_ml": float(prediction_volume),
        "signed_relative_volume_error": float(signed_relative_error),
        "absolute_relative_volume_error": float(abs(signed_relative_error)),
    }


def evaluate_segmentation_manifest(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    reference_column: str,
    prediction_column: str,
    bootstrap_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate patient-level masks and summarize each independent cohort."""

    required = [
        config.column("patient_id"),
        config.column("cohort"),
        reference_column,
        prediction_column,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Segmentation manifest is missing columns: {', '.join(missing)}")
    rows = []
    tolerance = float(config.section("segmentation")["surface_dice_tolerance_mm"])
    for record in frame.to_dict(orient="records"):
        reference_image = sitk.ReadImage(str(record[reference_column]), sitk.sitkUInt8)
        prediction_image = sitk.ReadImage(str(record[prediction_column]), sitk.sitkUInt8)
        if reference_image.GetSize() != prediction_image.GetSize():
            prediction_image = sitk.Resample(
                prediction_image,
                reference_image,
                sitk.Transform(3, sitk.sitkIdentity),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
        reference = sitk.GetArrayFromImage(reference_image) > 0
        prediction = sitk.GetArrayFromImage(prediction_image) > 0
        metrics = segmentation_metrics(
            reference,
            prediction,
            tuple(reversed(reference_image.GetSpacing())),
            surface_tolerance_mm=tolerance,
        )
        rows.append(
            {
                config.column("patient_id"): str(record[config.column("patient_id")]),
                config.column("cohort"): str(record[config.column("cohort")]),
                **metrics,
            }
        )
    patient_metrics = pd.DataFrame(rows)
    summary_rows = []
    metric_columns = [
        "dice",
        "surface_dice",
        "sensitivity",
        "hd95_mm",
        "absolute_relative_volume_error",
    ]
    rng = np.random.default_rng(seed)
    for cohort, group in patient_metrics.groupby(config.column("cohort"), sort=False):
        values = group[metric_columns].to_numpy(float)
        boot = np.empty((bootstrap_resamples, len(metric_columns)), dtype=float)
        for index in range(bootstrap_resamples):
            selected = rng.integers(0, len(group), size=len(group))
            boot[index] = np.mean(values[selected], axis=0)
        for metric_index, metric in enumerate(metric_columns):
            summary_rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "n": len(group),
                    "mean": float(np.mean(values[:, metric_index])),
                    "ci_low": float(np.percentile(boot[:, metric_index], 2.5)),
                    "ci_high": float(np.percentile(boot[:, metric_index], 97.5)),
                }
            )
    return patient_metrics, pd.DataFrame(summary_rows)


def safe_case_id(value: object) -> str:
    """Return a filesystem-safe case identifier."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
