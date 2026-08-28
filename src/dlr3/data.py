"""Manifest validation and imaging-path utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import StudyConfig

RAW_IMAGE_COLUMNS = (
    "t1_path",
    "t2_path",
    "flair_path",
    "ce_t1_path",
    "brain_mask_path",
    "tumor_mask_path",
)
PROCESSED_IMAGE_COLUMNS = (
    "preprocessed_t1_path",
    "preprocessed_t2_path",
    "preprocessed_flair_path",
    "preprocessed_ce_t1_path",
    "preprocessed_brain_mask_path",
    "preprocessed_tumor_mask_path",
    "voi_path",
)


@dataclass(frozen=True)
class ManifestAudit:
    """Summary returned after a manifest passes validation."""

    patients: int
    cohorts: dict[str, int]
    centers: dict[str, int]
    biological_subset: int


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a patient-level CSV manifest and resolve all path columns."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")
    frame = pd.read_csv(source)
    frame.attrs["source"] = source
    for column in (*RAW_IMAGE_COLUMNS, *PROCESSED_IMAGE_COLUMNS):
        if column in frame:
            frame[column] = frame[column].map(
                lambda value: _resolve_manifest_path(source.parent, value)
            )
    return frame


def _resolve_manifest_path(root: Path, value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def validate_manifest(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    require_paths: Iterable[str] = (),
    enforce_spatial_independence: bool = True,
) -> ManifestAudit:
    """Validate cohort isolation, endpoints, identifiers, and required paths."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    center_col = config.column("center_id")
    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    subset_col = config.column("biological_subset")
    required = [patient_col, cohort_col, center_col, time_col, event_col]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Manifest contains no patients.")
    if frame[patient_col].isna().any() or (frame[patient_col].astype(str).str.len() == 0).any():
        raise ValueError("Every row must contain a nonempty surrogate patient identifier.")
    duplicated = frame.loc[frame[patient_col].duplicated(keep=False), patient_col].astype(str)
    if not duplicated.empty:
        values = ", ".join(sorted(duplicated.unique())[:10])
        raise ValueError(f"Patient identifiers are not unique: {values}")
    unsafe = frame[patient_col].astype(str).map(_looks_like_direct_identifier)
    if unsafe.any():
        raise ValueError(
            "Patient identifiers must be nonidentifying surrogates without names, dates, or paths."
        )
    allowed_cohorts = {
        config.cohort("training"),
        config.cohort("temporal_validation"),
        config.cohort("spatial_validation"),
    }
    observed_cohorts = set(frame[cohort_col].dropna().astype(str))
    unknown = sorted(observed_cohorts.difference(allowed_cohorts))
    if unknown:
        raise ValueError(f"Unknown cohort labels: {', '.join(unknown)}")
    absent = sorted(allowed_cohorts.difference(observed_cohorts))
    if absent:
        raise ValueError(f"Required cohorts are absent: {', '.join(absent)}")
    time = pd.to_numeric(frame[time_col], errors="coerce")
    if time.isna().any() or (time <= 0).any():
        raise ValueError("Progression-free survival times must be finite and greater than zero.")
    event = pd.to_numeric(frame[event_col], errors="coerce")
    if event.isna().any() or not set(event.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Progression-free survival event values must be binary.")
    if subset_col in frame:
        subset = pd.to_numeric(frame[subset_col], errors="coerce").fillna(0).astype(int)
        if not set(subset.unique()).issubset({0, 1}):
            raise ValueError("Biological-subset membership must be binary.")
        invalid_subset = subset.eq(1) & frame[cohort_col].ne(config.cohort("training"))
        if invalid_subset.any():
            raise ValueError("The biological subset must be nested within the training cohort.")
    else:
        subset = pd.Series(0, index=frame.index, dtype=int)
    if enforce_spatial_independence:
        training_centers = set(
            frame.loc[frame[cohort_col].eq(config.cohort("training")), center_col].astype(str)
        )
        spatial_centers = set(
            frame.loc[frame[cohort_col].eq(config.cohort("spatial_validation")), center_col].astype(
                str
            )
        )
        overlap = sorted(training_centers.intersection(spatial_centers))
        if overlap:
            raise ValueError(
                "Spatial validation centers overlap with development centers: " + ", ".join(overlap)
            )
    required_paths = tuple(require_paths)
    absent_path_columns = [column for column in required_paths if column not in frame]
    if absent_path_columns:
        raise ValueError(f"Manifest is missing path columns: {', '.join(absent_path_columns)}")
    missing_files: list[str] = []
    for column in required_paths:
        for value in frame[column]:
            if not value or not Path(str(value)).is_file():
                missing_files.append(f"{column}: {value}")
                if len(missing_files) == 10:
                    break
        if len(missing_files) == 10:
            break
    if missing_files:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing_files))
    cohort_counts = frame[cohort_col].value_counts().sort_index().astype(int).to_dict()
    center_counts = frame.groupby(cohort_col)[center_col].nunique().astype(int).to_dict()
    return ManifestAudit(
        patients=len(frame),
        cohorts=cohort_counts,
        centers=center_counts,
        biological_subset=int(subset.sum()),
    )


def _looks_like_direct_identifier(value: str) -> bool:
    text = value.strip()
    if any(separator in text for separator in ("/", "\\", "@")):
        return True
    if re.search(r"\b(?:19|20)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}\b", text):
        return True
    return bool(re.search(r"\s{2,}", text))


def cohort_frame(frame: pd.DataFrame, config: StudyConfig, cohort_name: str) -> pd.DataFrame:
    """Return one configured cohort without changing row order."""

    label = config.cohort(cohort_name)
    return frame.loc[frame[config.column("cohort")].eq(label)].copy()
