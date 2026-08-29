"""Configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    for name in ("output_spacing_mm", "target_shape_dhw"):
        sequence = preprocessing.get(name)
        if (
            not isinstance(sequence, list)
            or len(sequence) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in sequence
            )
            or any(not math.isfinite(float(item)) or float(item) <= 0 for item in sequence)
        ):
            raise ValueError(f"Preprocessing value '{name}' must contain three positive numbers.")
    segmentation = values["segmentation"]
    if segmentation.get("folds") != [0, 1, 2, 3, 4]:
        raise ValueError("The locked nnU-Net folds must be 0, 1, 2, 3, and 4.")
    if values["deep_survival"].get("input_channels") != 4:
        raise ValueError("Deep survival models require exactly four input channels.")
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
    calibration_horizon = values["evaluation"].get("calibration_horizon_months")
    if calibration_horizon not in horizons:
        raise ValueError("The calibration horizon must be one of the ICVS prediction horizons.")
    return StudyConfig(values=values, source=source)
