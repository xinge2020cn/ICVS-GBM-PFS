from pathlib import Path

import yaml

from icvs_gbm_pfs.config import load_config

CONFIG_DIR = Path(__file__).parents[1] / "configs"


def test_locked_imaging_and_segmentation_protocol() -> None:
    config = load_config(CONFIG_DIR / "study.yaml")
    assert config.values["modalities"] == ["t1", "t2", "flair", "ce_t1"]
    preprocessing = config.section("preprocessing")
    assert preprocessing == {
        "reference_modality": "ce_t1",
        "output_spacing_mm": [1.0, 1.0, 5.0],
        "peritumoral_margin_mm": 10.0,
        "target_shape_dhw": [24, 192, 192],
    }
    segmentation = config.section("segmentation")
    assert segmentation["implementation_version"] == "2.4.2"
    assert segmentation["configuration"] == "3d_fullres"
    assert segmentation["folds"] == [0, 1, 2, 3, 4]
    assert segmentation["patch_shape_dhw"] == [32, 192, 192]
    assert segmentation["epochs"] == 1000
    assert segmentation["training_iterations_per_epoch"] == 250
    assert segmentation["validation_iterations_per_epoch"] == 50
    assert segmentation["batch_size"] == 2
    assert segmentation["initial_learning_rate"] == 0.01
    assert segmentation["nesterov_momentum"] == 0.99
    assert segmentation["weight_decay"] == 0.00003
    assert segmentation["foreground_oversampling_fraction"] == 0.33
    assert segmentation["deep_supervision"] is True
    assert segmentation["automatic_mixed_precision"] is True
    assert segmentation["surface_dice_tolerance_mm"] == 2.0


def test_locked_radiomics_protocol() -> None:
    parameters = yaml.safe_load((CONFIG_DIR / "radiomics.yaml").read_text(encoding="utf-8"))
    assert parameters["imageType"] == {
        "Original": {},
        "Wavelet": {"wavelet": "coif1"},
        "LoG": {"sigma": [1.0, 2.0, 3.0]},
    }
    assert set(parameters["featureClass"]) == {
        "firstorder",
        "shape",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
    }
    assert parameters["setting"] == {
        "binWidth": 0.25,
        "normalize": False,
        "correctMask": True,
        "force2D": False,
    }


def test_locked_deep_survival_and_icvs_protocol() -> None:
    config = load_config(CONFIG_DIR / "study.yaml")
    deep = config.section("deep_survival")
    vit = deep["vit"]
    assert vit["patch_shape_dhw"] == [4, 16, 16]
    assert (24 // 4) * (192 // 16) * (192 // 16) == 864
    assert (vit["embedding_dim"], vit["depth"], vit["heads"], vit["mlp_ratio"]) == (
        256,
        6,
        8,
        4.0,
    )
    assert (vit["attention_dropout"], vit["projection_dropout"]) == (0.10, 0.10)
    assert (vit["head_dim"], vit["head_dropout"], vit["epochs"]) == (64, 0.30, 120)
    assert vit["physical_batch_size"] * vit["accumulation_steps"] == 16
    resnet = deep["resnet"]
    assert resnet["stage_blocks"] == [2, 2, 2, 2]
    assert resnet["stage_channels"] == [64, 128, 256, 512]
    assert (resnet["head_dim"], resnet["head_dropout"], resnet["epochs"]) == (
        64,
        0.30,
        100,
    )
    assert resnet["physical_batch_size"] * resnet["accumulation_steps"] == 16
    optimization = deep["optimization"]
    assert optimization == {
        "learning_rate": 0.0001,
        "weight_decay": 0.00001,
        "warmup_epochs": 10,
        "minimum_learning_rate": 0.000001,
        "gradient_clip_norm": 1.0,
        "tuning_fraction": 0.20,
        "maximum_tuning_epochs": 200,
        "early_stopping_patience": 30,
        "crossfit_folds": 5,
    }
    assert config.section("icvs") == {
        "horizons_months": [6, 12, 18, 24, 30, 36],
        "n_estimators": 500,
        "max_features": 2,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
    }
    assert config.section("evaluation") == {
        "bootstrap_resamples": 1000,
        "calibration_horizon_months": 12,
        "calibration_groups": 8,
    }
