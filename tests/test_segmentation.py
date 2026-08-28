import numpy as np
import pytest

pytest.importorskip("SimpleITK")

from icvs_gbm_pfs.segmentation import segmentation_metrics


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
