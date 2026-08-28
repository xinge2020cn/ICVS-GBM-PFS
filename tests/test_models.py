import pytest

torch = pytest.importorskip("torch")

from icvs_gbm_pfs.models import ResNet3DSurvival, VisionTransformer3DSurvival  # noqa: E402


def test_vit_returns_one_log_risk_per_volume() -> None:
    model = VisionTransformer3DSurvival(
        input_shape_dhw=(8, 32, 32),
        patch_shape_dhw=(4, 16, 16),
        embedding_dim=32,
        depth=2,
        heads=4,
        head_dim=16,
    )
    output = model(torch.zeros(2, 4, 8, 32, 32))
    assert output.shape == (2,)


def test_resnet_returns_one_log_risk_per_volume() -> None:
    model = ResNet3DSurvival(
        stage_blocks=(1, 1, 1, 1),
        stage_channels=(8, 16, 32, 64),
        head_dim=16,
    )
    model.eval()
    output = model(torch.zeros(2, 4, 8, 32, 32))
    assert output.shape == (2,)
