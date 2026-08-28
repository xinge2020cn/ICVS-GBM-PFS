"""Configuration loading and validation."""

from __future__ import annotations

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
    required = {"seed", "columns", "cohorts", "modalities", "preprocessing"}
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    return StudyConfig(values=values, source=source)
