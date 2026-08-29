from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("SimpleITK")
pytest.importorskip("torch")
import SimpleITK as sitk  # noqa: E402

from icvs_gbm_pfs.preprocessing import (  # noqa: E402
    build_tumor_peritumoral_voi,
    load_cropped_volume,
)


def test_peritumoral_margin_is_confined_to_the_axial_plane() -> None:
    tumor = sitk.Image([21, 21, 3], sitk.sitkUInt8)
    tumor.SetSpacing((1.0, 1.0, 5.0))
    tumor[10, 10, 1] = 1
    brain = sitk.Image([21, 21, 3], sitk.sitkUInt8) + 1
    brain.CopyInformation(tumor)
    voi = build_tumor_peritumoral_voi(tumor, brain, 2.0)
    array = sitk.GetArrayFromImage(voi)
    assert array[1].sum() > 1
    assert array[0].sum() == 0
    assert array[2].sum() == 0


def test_volume_crop_has_locked_network_geometry(tmp_path: Path) -> None:
    paths = {}
    for modality_index, modality in enumerate(("t1", "t2", "flair", "ce_t1"), start=1):
        array = np.zeros((6, 12, 12), dtype=np.float32)
        array[2:5, 3:9, 4:10] = float(modality_index)
        image = sitk.GetImageFromArray(array)
        path = tmp_path / f"{modality}.nii.gz"
        sitk.WriteImage(image, str(path))
        paths[f"preprocessed_{modality}_path"] = str(path)
    mask_array = np.zeros((6, 12, 12), dtype=np.uint8)
    mask_array[2:5, 3:9, 4:10] = 1
    mask = sitk.GetImageFromArray(mask_array)
    mask_path = tmp_path / "voi.nii.gz"
    sitk.WriteImage(mask, str(mask_path))
    volume, resized_mask = load_cropped_volume({**paths, "voi_path": str(mask_path)}, (8, 32, 32))
    assert volume.shape == (4, 8, 32, 32)
    assert resized_mask.shape == (1, 8, 32, 32)
    assert volume[3].max().item() == pytest.approx(4.0)


def test_volume_crop_rejects_mismatched_physical_geometry(tmp_path: Path) -> None:
    paths = {}
    for modality in ("t1", "t2", "flair", "ce_t1"):
        image = sitk.GetImageFromArray(np.ones((4, 8, 8), dtype=np.float32))
        if modality == "t2":
            image.SetOrigin((1.0, 0.0, 0.0))
        path = tmp_path / f"{modality}.nii.gz"
        sitk.WriteImage(image, str(path))
        paths[f"preprocessed_{modality}_path"] = str(path)
    mask = sitk.GetImageFromArray(np.ones((4, 8, 8), dtype=np.uint8))
    mask_path = tmp_path / "voi.nii.gz"
    sitk.WriteImage(mask, str(mask_path))
    with pytest.raises(ValueError, match="identical physical geometry"):
        load_cropped_volume({**paths, "voi_path": str(mask_path)}, (4, 8, 8))
