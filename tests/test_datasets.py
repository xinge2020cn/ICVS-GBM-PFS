import pytest

torch = pytest.importorskip("torch")

from icvs_gbm_pfs.datasets import JointMRITransform  # noqa: E402


def test_joint_augmentation_preserves_zero_background() -> None:
    image = torch.zeros(4, 8, 32, 32)
    mask = torch.zeros(1, 8, 32, 32)
    image[:, 2:6, 8:24, 8:24] = 1.0
    mask[:, 2:6, 8:24, 8:24] = 1.0
    transformed = JointMRITransform(seed=2026)(image, mask)
    assert torch.count_nonzero(transformed[:, :, :2, :2]) == 0
    assert torch.isfinite(transformed).all()


def test_gaussian_smoothing_preserves_shape_and_channel_independence() -> None:
    image = torch.zeros(2, 7, 7, 7)
    image[0, 3, 3, 3] = 1.0
    result = JointMRITransform._gaussian_smooth(image, 0.75)
    assert result.shape == image.shape
    assert result[0, 3, 3, 3] < 1.0
    assert result[0].sum() == pytest.approx(1.0, rel=1e-5)
    assert torch.count_nonzero(result[1]) == 0
