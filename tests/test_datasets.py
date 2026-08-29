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
