"""MRI registration, bias correction, normalization, and VOI construction."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as functional

from .config import StudyConfig
from .data import RAW_IMAGE_COLUMNS, read_manifest, validate_manifest

MODALITY_COLUMNS = {
    "t1": "t1_path",
    "t2": "t2_path",
    "flair": "flair_path",
    "ce_t1": "ce_t1_path",
}


def _same_image_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    """Return whether two images occupy the same physical voxel grid."""

    return (
        first.GetDimension() == second.GetDimension()
        and first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), rtol=0.0, atol=1e-6)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), rtol=0.0, atol=1e-6)
        and np.allclose(first.GetDirection(), second.GetDirection(), rtol=0.0, atol=1e-6)
    )


def rigid_register(moving: sitk.Image, fixed: sitk.Image, *, seed: int) -> sitk.Image:
    """Rigidly register one MRI sequence to the anatomical reference."""

    moving_float = sitk.Cast(moving, sitk.sitkFloat32)
    fixed_float = sitk.Cast(fixed, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed_float,
        moving_float,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.20, seed=int(seed))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = registration.Execute(fixed_float, moving_float)
    return sitk.Resample(
        moving_float,
        fixed_float,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def n4_bias_correct(image: sitk.Image, brain_mask: sitk.Image) -> sitk.Image:
    """Apply N4 bias-field correction inside the brain mask."""

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(1e-7)
    corrected = corrector.Execute(
        sitk.Cast(image, sitk.sitkFloat32), sitk.Cast(brain_mask > 0, sitk.sitkUInt8)
    )
    return sitk.Cast(corrected, sitk.sitkFloat32)


def make_resampled_reference(
    image: sitk.Image,
    spacing_xyz: tuple[float, float, float],
) -> sitk.Image:
    """Create an empty reference image with the requested physical spacing."""

    if len(spacing_xyz) != 3 or not np.isfinite(spacing_xyz).all() or any(
        value <= 0 for value in spacing_xyz
    ):
        raise ValueError("Output spacing must contain three finite positive values.")
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    size = tuple(
        max(1, int(round(original_size[index] * original_spacing[index] / spacing_xyz[index])))
        for index in range(3)
    )
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing(spacing_xyz)
    return reference


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    *,
    is_mask: bool,
) -> sitk.Image:
    """Resample an image or mask into a common reference grid."""

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        interpolator,
        0.0,
        pixel_type,
    )


def zscore_within_mask(image: sitk.Image, brain_mask: sitk.Image) -> sitk.Image:
    """Standardize nonzero brain voxels for one patient and sequence."""

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    mask = sitk.GetArrayFromImage(brain_mask) > 0
    values = array[mask]
    if values.size == 0:
        raise ValueError("Brain mask contains no foreground voxels.")
    standard_deviation = float(values.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("MRI intensities have zero or invalid variance inside the brain mask.")
    normalized = np.zeros_like(array, dtype=np.float32)
    normalized[mask] = (values - float(values.mean())) / standard_deviation
    result = sitk.GetImageFromArray(normalized)
    result.CopyInformation(image)
    return result


def build_tumor_peritumoral_voi(
    tumor_mask: sitk.Image,
    brain_mask: sitk.Image,
    margin_mm: float,
) -> sitk.Image:
    """Combine the tumor core with an in-plane outward margin constrained to brain."""

    if not np.isfinite(margin_mm) or margin_mm <= 0:
        raise ValueError("The peritumoral margin must be finite and greater than zero.")
    if not _same_image_geometry(tumor_mask, brain_mask):
        raise ValueError("Tumor and brain masks must use identical physical geometry.")
    spacing = tumor_mask.GetSpacing()
    if not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError("Mask spacing must contain finite positive values.")
    radius = [
        max(1, int(math.ceil(margin_mm / spacing[0]))),
        max(1, int(math.ceil(margin_mm / spacing[1]))),
        0,
    ]
    tumor = sitk.Cast(tumor_mask > 0, sitk.sitkUInt8)
    brain = sitk.Cast(brain_mask > 0, sitk.sitkUInt8)
    dilated = sitk.BinaryDilate(tumor, radius, sitk.sitkBall, 0, 1, False)
    voi = sitk.And(dilated, brain)
    voi = sitk.Or(voi, tumor)
    return sitk.Cast(voi, sitk.sitkUInt8)


def process_subject(
    row: Mapping[str, object],
    output_root: str | Path,
    *,
    patient_id_column: str,
    spacing_xyz: tuple[float, float, float],
    margin_mm: float,
    seed: int,
) -> dict[str, str]:
    """Preprocess all MRI sequences and write one patient-level output set."""

    patient_id = str(row[patient_id_column])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id)
    patient_dir = Path(output_root).resolve() / safe_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    images = {
        modality: sitk.ReadImage(str(row[column]), sitk.sitkFloat32)
        for modality, column in MODALITY_COLUMNS.items()
    }
    reference = images["ce_t1"]
    brain_mask = sitk.ReadImage(str(row["brain_mask_path"]), sitk.sitkUInt8)
    tumor_mask = sitk.ReadImage(str(row["tumor_mask_path"]), sitk.sitkUInt8)
    if not _same_image_geometry(brain_mask, reference):
        brain_mask = resample_to_reference(brain_mask, reference, is_mask=True)
    if not _same_image_geometry(tumor_mask, reference):
        tumor_mask = resample_to_reference(tumor_mask, reference, is_mask=True)
    if not np.any(sitk.GetArrayViewFromImage(brain_mask)):
        raise ValueError(f"Brain mask contains no foreground voxels for patient {patient_id}.")
    if not np.any(sitk.GetArrayViewFromImage(tumor_mask)):
        raise ValueError(f"Tumor mask contains no foreground voxels for patient {patient_id}.")
    registered = {"ce_t1": reference}
    for modality in ("t1", "t2", "flair"):
        registered[modality] = rigid_register(images[modality], reference, seed=seed)
    corrected = {
        modality: n4_bias_correct(image, brain_mask) for modality, image in registered.items()
    }
    target_reference = make_resampled_reference(reference, spacing_xyz)
    brain_resampled = resample_to_reference(brain_mask, target_reference, is_mask=True)
    tumor_resampled = resample_to_reference(tumor_mask, target_reference, is_mask=True)
    outputs: dict[str, str] = {}
    for modality in ("t1", "t2", "flair", "ce_t1"):
        image = resample_to_reference(corrected[modality], target_reference, is_mask=False)
        image = zscore_within_mask(image, brain_resampled)
        path = patient_dir / f"{safe_id}_{modality}.nii.gz"
        sitk.WriteImage(image, str(path), True)
        outputs[f"preprocessed_{modality}_path"] = str(path)
    brain_path = patient_dir / f"{safe_id}_brain_mask.nii.gz"
    tumor_path = patient_dir / f"{safe_id}_tumor_core.nii.gz"
    voi_path = patient_dir / f"{safe_id}_tumor_peritumoral_voi.nii.gz"
    sitk.WriteImage(brain_resampled, str(brain_path), True)
    sitk.WriteImage(tumor_resampled, str(tumor_path), True)
    voi = build_tumor_peritumoral_voi(tumor_resampled, brain_resampled, margin_mm)
    sitk.WriteImage(voi, str(voi_path), True)
    outputs["preprocessed_brain_mask_path"] = str(brain_path)
    outputs["preprocessed_tumor_mask_path"] = str(tumor_path)
    outputs["voi_path"] = str(voi_path)
    return outputs


def preprocess_manifest(
    manifest_path: str | Path,
    config: StudyConfig,
    output_root: str | Path,
    output_manifest: str | Path,
) -> pd.DataFrame:
    """Preprocess every subject and create a manifest containing derived paths."""

    frame = read_manifest(manifest_path, patient_id_column=config.column("patient_id"))
    validate_manifest(frame, config, require_paths=RAW_IMAGE_COLUMNS)
    settings = config.section("preprocessing")
    spacing = tuple(float(value) for value in settings["output_spacing_mm"])
    if len(spacing) != 3 or not np.isfinite(spacing).all() or any(value <= 0 for value in spacing):
        raise ValueError(
            "Output spacing must contain three finite positive values in x, y, z order."
        )
    margin = float(settings["peritumoral_margin_mm"])
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("The peritumoral margin must be finite and greater than zero.")
    derived = []
    for row in frame.to_dict(orient="records"):
        derived.append(
            process_subject(
                row,
                output_root,
                patient_id_column=config.column("patient_id"),
                spacing_xyz=spacing,
                margin_mm=margin,
                seed=config.seed,
            )
        )
    result = pd.concat([frame.reset_index(drop=True), pd.DataFrame(derived)], axis=1)
    destination = Path(output_manifest).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def load_cropped_volume(
    row: Mapping[str, object],
    target_shape_dhw: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the common VOI crop and resize it to the network input geometry."""

    target_shape = tuple(int(value) for value in target_shape_dhw)
    if len(target_shape) != 3 or any(value <= 0 for value in target_shape):
        raise ValueError("Target volume shape must contain three positive dimensions.")
    arrays = []
    reference_image = None
    for modality in ("t1", "t2", "flair", "ce_t1"):
        image = sitk.ReadImage(str(row[f"preprocessed_{modality}_path"]), sitk.sitkFloat32)
        if reference_image is None:
            reference_image = image
        elif not _same_image_geometry(image, reference_image):
            raise ValueError("Preprocessed MRI volumes must use identical physical geometry.")
        array = sitk.GetArrayFromImage(image).astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Preprocessed MRI volumes must contain only finite intensities.")
        arrays.append(array)
    mask_image = sitk.ReadImage(str(row["voi_path"]), sitk.sitkUInt8)
    if reference_image is None or not _same_image_geometry(mask_image, reference_image):
        raise ValueError("The VOI mask and preprocessed MRI volumes must use identical geometry.")
    mask = sitk.GetArrayFromImage(mask_image) > 0
    if not mask.any():
        raise ValueError("Tumor-peritumoral VOI contains no foreground voxels.")
    coordinates = np.where(mask)
    lower = np.array([axis.min() for axis in coordinates], dtype=int)
    upper = np.array([axis.max() + 1 for axis in coordinates], dtype=int)
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    volume = np.stack([array[slices] for array in arrays], axis=0)
    mask_crop = mask[slices].astype(np.float32)
    tensor = torch.from_numpy(volume).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_crop).unsqueeze(0).unsqueeze(0)
    tensor = functional.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)
    mask_tensor = functional.interpolate(mask_tensor, size=target_shape, mode="nearest")
    mask_tensor = mask_tensor.squeeze(0)
    tensor = tensor * mask_tensor
    return tensor.contiguous(), mask_tensor.contiguous()
