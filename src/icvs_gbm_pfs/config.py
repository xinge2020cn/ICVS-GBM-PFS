"""Configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _positive_integer(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer greater than or equal to {minimum}.")
    return value


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = False,
    maximum_inclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    if minimum is not None and (
        result < minimum if minimum_inclusive else result <= minimum
    ):
        qualifier = "greater than or equal to" if minimum_inclusive else "greater than"
        raise ValueError(f"{label} must be {qualifier} {minimum}.")
    if maximum is not None and (
        result > maximum if maximum_inclusive else result >= maximum
    ):
        qualifier = "less than or equal to" if maximum_inclusive else "less than"
        raise ValueError(f"{label} must be {qualifier} {maximum}.")
    return result


def _positive_integer_sequence(value: object, label: str, *, length: int) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} positive integers.")
    for item in value:
        _positive_integer(item, label)
    return value


@dataclass(frozen=True)
class StudyConfig:
    """Read-only access to the study configuration."""

    values: dict[str, Any]
    source: Path

    @property
    def seed(self) -> int:
        return int(self.values["seed"])

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Configuration section '{name}' is missing or invalid.")
        return value

    def column(self, name: str) -> str:
        value = self.section("columns").get(name)
        if not isinstance(value, str) or not value:
            raise KeyError(f"Column mapping '{name}' is missing or invalid.")
        return value

    def cohort(self, name: str) -> str:
        value = self.section("cohorts").get(name)
        if not isinstance(value, str) or not value:
            raise KeyError(f"Cohort mapping '{name}' is missing or invalid.")
        return value


def load_config(path: str | Path) -> StudyConfig:
    """Load a YAML study configuration."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    with source.open("r", encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("The configuration root must be a mapping.")
    required = {
        "seed",
        "columns",
        "cohorts",
        "modalities",
        "preprocessing",
        "segmentation",
        "deep_survival",
        "radiomics",
        "icvs",
        "evaluation",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    if (
        isinstance(values["seed"], bool)
        or not isinstance(values["seed"], int)
        or not 0 <= values["seed"] < 2**32
    ):
        raise ValueError("The reproducibility seed must be an integer from 0 to 2^32 - 1.")
    for section in required.difference({"seed", "modalities"}):
        if not isinstance(values[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")
    required_columns = {
        "patient_id",
        "cohort",
        "center_id",
        "pfs_time",
        "pfs_event",
        "biological_subset",
        "age",
        "mgmt",
        "extent_of_resection",
    }
    columns = values["columns"]
    if not isinstance(columns, dict):
        raise ValueError("The column configuration must be a mapping.")
    missing_columns = sorted(required_columns.difference(columns))
    if missing_columns:
        raise ValueError(f"Missing column mappings: {', '.join(missing_columns)}")
    column_names = [columns[name] for name in sorted(required_columns)]
    if any(not isinstance(name, str) or not name for name in column_names):
        raise ValueError("Every required column mapping must be a nonempty string.")
    if len(set(column_names)) != len(column_names):
        raise ValueError("Required manifest column mappings must be unique.")
    required_cohorts = {"training", "temporal_validation", "spatial_validation"}
    cohorts = values["cohorts"]
    if not isinstance(cohorts, dict):
        raise ValueError("The cohort configuration must be a mapping.")
    missing_cohorts = sorted(required_cohorts.difference(cohorts))
    if missing_cohorts:
        raise ValueError(f"Missing cohort mappings: {', '.join(missing_cohorts)}")
    cohort_names = [cohorts[name] for name in sorted(required_cohorts)]
    if any(not isinstance(name, str) or not name for name in cohort_names):
        raise ValueError("Every required cohort mapping must be a nonempty string.")
    if len(set(cohort_names)) != len(cohort_names):
        raise ValueError("Training, temporal, and spatial cohort labels must be distinct.")
    if values["modalities"] != ["t1", "t2", "flair", "ce_t1"]:
        raise ValueError("Modalities must be ordered as t1, t2, flair, and ce_t1.")
    preprocessing = values["preprocessing"]
    if preprocessing.get("reference_modality") != "ce_t1":
        raise ValueError("The locked preprocessing reference modality must be ce_t1.")
    spacing = preprocessing.get("output_spacing_mm")
    if (
        not isinstance(spacing, list)
        or len(spacing) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in spacing
        )
        or any(not math.isfinite(float(item)) or float(item) <= 0 for item in spacing)
    ):
        raise ValueError("The preprocessing output spacing must contain three positive numbers.")
    target_shape = _positive_integer_sequence(
        preprocessing.get("target_shape_dhw"), "The network input shape", length=3
    )
    _finite_number(
        preprocessing.get("peritumoral_margin_mm"),
        "The peritumoral margin",
        minimum=0.0,
    )
    segmentation = values["segmentation"]
    _positive_integer(segmentation.get("dataset_id"), "The nnU-Net dataset identifier")
    if not isinstance(segmentation.get("configuration"), str) or not segmentation[
        "configuration"
    ].strip():
        raise ValueError("The nnU-Net configuration must be a nonempty string.")
    if segmentation.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("The locked nnU-Net folds must be 0, 1, 2, 3, and 4.")
    _positive_integer_sequence(
        segmentation.get("patch_shape_dhw"), "The nnU-Net patch shape", length=3
    )
    _positive_integer(segmentation.get("epochs"), "The nnU-Net epoch count")
    _positive_integer(segmentation.get("batch_size"), "The nnU-Net batch size")
    _finite_number(
        segmentation.get("surface_dice_tolerance_mm"),
        "The surface Dice tolerance",
        minimum=0.0,
    )
    deep_survival = values["deep_survival"]
    if deep_survival.get("input_channels") != 4:
        raise ValueError("Deep survival models require exactly four input channels.")
    for section in ("vit", "resnet", "optimization"):
        if not isinstance(deep_survival.get(section), dict):
            raise ValueError(f"Deep-survival section '{section}' must be a mapping.")
    vit = deep_survival["vit"]
    patch_shape = _positive_integer_sequence(
        vit.get("patch_shape_dhw"), "The ViT patch shape", length=3
    )
    if any(size % patch != 0 for size, patch in zip(target_shape, patch_shape, strict=True)):
        raise ValueError("Every network input dimension must be divisible by its ViT patch size.")
    embedding_dim = _positive_integer(vit.get("embedding_dim"), "The ViT embedding dimension")
    heads = _positive_integer(vit.get("heads"), "The ViT attention-head count")
    if embedding_dim % heads != 0:
        raise ValueError("The ViT embedding dimension must be divisible by the attention heads.")
    for name in ("depth", "head_dim", "epochs", "physical_batch_size", "accumulation_steps"):
        _positive_integer(vit.get(name), f"The ViT {name.replace('_', ' ')}")
    _finite_number(vit.get("mlp_ratio"), "The ViT MLP ratio", minimum=0.0)
    for name in ("attention_dropout", "projection_dropout", "head_dropout"):
        _finite_number(
            vit.get(name),
            f"The ViT {name.replace('_', ' ')}",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=True,
        )
    resnet = deep_survival["resnet"]
    _positive_integer_sequence(
        resnet.get("stage_blocks"), "The ResNet stage block counts", length=4
    )
    _positive_integer_sequence(
        resnet.get("stage_channels"), "The ResNet stage channel counts", length=4
    )
    for name in ("head_dim", "epochs", "physical_batch_size", "accumulation_steps"):
        _positive_integer(resnet.get(name), f"The ResNet {name.replace('_', ' ')}")
    _finite_number(
        resnet.get("head_dropout"),
        "The ResNet head dropout",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=True,
    )
    optimization = deep_survival["optimization"]
    learning_rate = _finite_number(
        optimization.get("learning_rate"), "The learning rate", minimum=0.0
    )
    minimum_learning_rate = _finite_number(
        optimization.get("minimum_learning_rate"),
        "The minimum learning rate",
        minimum=0.0,
    )
    if minimum_learning_rate > learning_rate:
        raise ValueError("The minimum learning rate cannot exceed the initial learning rate.")
    _finite_number(
        optimization.get("weight_decay"),
        "The weight decay",
        minimum=0.0,
        minimum_inclusive=True,
    )
    _finite_number(
        optimization.get("gradient_clip_norm"),
        "The gradient-clipping norm",
        minimum=0.0,
    )
    _finite_number(
        optimization.get("tuning_fraction"),
        "The tuning fraction",
        minimum=0.0,
        maximum=1.0,
    )
    _positive_integer(
        optimization.get("warmup_epochs"), "The warmup epoch count", minimum=0
    )
    _positive_integer(
        optimization.get("maximum_tuning_epochs"), "The maximum tuning epoch count"
    )
    _positive_integer(
        optimization.get("early_stopping_patience"), "The early-stopping patience"
    )
    _positive_integer(
        optimization.get("crossfit_folds"), "The deep-survival cross-fitting fold count", minimum=2
    )
    radiomics = values["radiomics"]
    _finite_number(
        radiomics.get("univariable_p_threshold"),
        "The radiomics univariable P-value threshold",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=True,
    )
    _positive_integer(
        radiomics.get("cross_validation_folds"),
        "The radiomics cross-validation fold count",
        minimum=2,
    )
    if not isinstance(radiomics.get("use_one_standard_error_rule"), bool):
        raise ValueError("The radiomics one-standard-error setting must be Boolean.")
    horizons = values["icvs"].get("horizons_months")
    if (
        not isinstance(horizons, list)
        or len(horizons) < 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in horizons)
        or any(not math.isfinite(float(item)) or float(item) <= 0 for item in horizons)
        or any(float(item) != int(item) for item in horizons)
        or any(
            float(horizons[index]) >= float(horizons[index + 1])
            for index in range(len(horizons) - 1)
        )
    ):
        raise ValueError("ICVS horizons must be increasing positive whole-month values.")
    icvs = values["icvs"]
    _positive_integer(icvs.get("n_estimators"), "The ICVS estimator count")
    max_features = _positive_integer(icvs.get("max_features"), "The ICVS max-features value")
    if max_features > 4:
        raise ValueError("The ICVS max-features value cannot exceed its four predictors.")
    _positive_integer(
        icvs.get("min_samples_split"), "The ICVS minimum split size", minimum=2
    )
    _positive_integer(icvs.get("min_samples_leaf"), "The ICVS minimum leaf size")
    evaluation = values["evaluation"]
    _positive_integer(
        evaluation.get("bootstrap_resamples"),
        "The evaluation bootstrap resample count",
        minimum=100,
    )
    _positive_integer(
        evaluation.get("calibration_groups"), "The calibration group count", minimum=2
    )
    calibration_horizon = evaluation.get("calibration_horizon_months")
    if calibration_horizon not in horizons:
        raise ValueError("The calibration horizon must be one of the ICVS prediction horizons.")
    return StudyConfig(values=values, source=source)
