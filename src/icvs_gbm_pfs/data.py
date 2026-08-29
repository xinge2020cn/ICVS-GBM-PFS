"""Manifest validation and imaging-path utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


def read_manifest(
    path: str | Path,
    *,
    patient_id_column: str = "patient_id",
) -> pd.DataFrame:
    """Read a patient-level CSV manifest and resolve all path columns."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")
    frame = pd.read_csv(source, dtype={patient_id_column: "string"})
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
    age_col = config.column("age")
    sex_col = config.column("sex")
    location_col = config.column("tumor_location")
    laterality_col = config.column("laterality")
    mgmt_col = config.column("mgmt")
    resection_col = config.column("extent_of_resection")
    treatment_col = config.column("postoperative_treatment")
    required = [
        patient_col,
        cohort_col,
        center_col,
        time_col,
        event_col,
        subset_col,
        age_col,
        sex_col,
        location_col,
        laterality_col,
        mgmt_col,
        resection_col,
        treatment_col,
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Manifest contains no patients.")
    patient_ids = frame[patient_col].astype("string")
    if patient_ids.isna().any() or patient_ids.str.strip().eq("").any():
        raise ValueError("Every row must contain a nonempty surrogate patient identifier.")
    if not patient_ids.str.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*").all():
        raise ValueError(
            "Patient identifiers may contain only ASCII letters, numbers, periods, underscores, "
            "and hyphens, and must start with a letter or number."
        )
    if patient_ids.str.endswith(".").any():
        raise ValueError("Patient identifiers must not end with a period.")
    reserved_names = {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *{f"com{index}" for index in range(1, 10)},
        *{f"lpt{index}" for index in range(1, 10)},
    }
    if patient_ids.str.split(".").str[0].str.casefold().isin(reserved_names).any():
        raise ValueError("Patient identifiers must not use reserved filesystem names.")
    normalized_ids = patient_ids.str.casefold()
    duplicated = frame.loc[normalized_ids.duplicated(keep=False), patient_col].astype(str)
    if not duplicated.empty:
        values = ", ".join(sorted(duplicated.unique())[:10])
        raise ValueError(
            "Patient identifiers are not unique under case-insensitive matching: " + values
        )
    unsafe = patient_ids.map(_looks_like_direct_identifier)
    if unsafe.any():
        raise ValueError(
            "Patient identifiers must be nonidentifying surrogates without names, dates, or paths."
        )
    allowed_cohorts = {
        config.cohort("training"),
        config.cohort("temporal_validation"),
        config.cohort("spatial_validation"),
    }
    cohort_labels = frame[cohort_col].astype("string")
    if cohort_labels.isna().any() or cohort_labels.str.strip().eq("").any():
        raise ValueError("Every row must contain a cohort label.")
    observed_cohorts = set(cohort_labels.astype(str))
    unknown = sorted(observed_cohorts.difference(allowed_cohorts))
    if unknown:
        raise ValueError(f"Unknown cohort labels: {', '.join(unknown)}")
    absent = sorted(allowed_cohorts.difference(observed_cohorts))
    if absent:
        raise ValueError(f"Required cohorts are absent: {', '.join(absent)}")
    centers = frame[center_col].astype("string")
    if centers.isna().any() or centers.str.strip().eq("").any():
        raise ValueError("Every row must contain a nonempty center identifier.")
    time = pd.to_numeric(frame[time_col], errors="coerce")
    if not np.isfinite(time.to_numpy(float)).all() or (time <= 0).any():
        raise ValueError("Progression-free survival times must be finite and greater than zero.")
    event = pd.to_numeric(frame[event_col], errors="coerce")
    if event.isna().any() or not event.isin([0, 1]).all():
        raise ValueError("Progression-free survival event values must be binary.")
    subset = pd.to_numeric(frame[subset_col], errors="coerce")
    if subset.isna().any() or not subset.isin([0, 1]).all():
        raise ValueError("Biological-subset membership must be binary.")
    invalid_subset = subset.eq(1) & frame[cohort_col].ne(config.cohort("training"))
    if invalid_subset.any():
        raise ValueError("The biological subset must be nested within the training cohort.")
    age = pd.to_numeric(frame[age_col], errors="coerce")
    if not np.isfinite(age.to_numpy(float)).all() or (age < 18).any():
        raise ValueError("Age values must be finite and at least 18 years.")
    for column, label in (
        (mgmt_col, "MGMT promoter methylation"),
        (resection_col, "Extent of resection"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"{label} values must use binary zero-one coding.")
    categorical_fields = (
        (sex_col, "Sex", {"Female", "Male"}),
        (
            location_col,
            "Tumor location",
            {"Frontal", "Temporal", "Parietal", "Occipital", "Deep-seated"},
        ),
        (laterality_col, "Laterality", {"Left", "Right", "Midline/bilateral"}),
        (
            treatment_col,
            "Postoperative treatment",
            {"Stupp regimen", "Radiotherapy only", "Temozolomide only", "Other/none"},
        ),
    )
    for column, label, allowed in categorical_fields:
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{label} values must be complete.")
        unknown_values = sorted(set(values.astype(str)).difference(allowed))
        if unknown_values:
            raise ValueError(
                f"{label} contains unsupported values: {', '.join(unknown_values)}"
            )
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
