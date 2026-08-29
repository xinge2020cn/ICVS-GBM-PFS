"""Preparation of the nested transcriptomic cohort from cross-fitted ViT scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StudyConfig


def prepare_biological_cohort(
    manifest: pd.DataFrame,
    vit_scores: pd.DataFrame,
    config: StudyConfig,
) -> pd.DataFrame:
    """Create the locked biological subset using training out-of-fold ViT scores."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    subset_col = config.column("biological_subset")
    required = [patient_col, "vit_score_oof"]
    missing = [column for column in required if column not in vit_scores]
    if missing:
        raise ValueError(f"ViT score table is missing columns: {', '.join(missing)}")
    if vit_scores[patient_col].duplicated().any():
        raise ValueError("ViT score table contains duplicate patient identifiers.")
    if set(manifest[patient_col].astype(str)) != set(vit_scores[patient_col].astype(str)):
        raise ValueError("ViT scores must contain exactly one row for every manifest patient.")
    frame = manifest.merge(
        vit_scores[[patient_col, "vit_score_oof"]],
        on=patient_col,
        how="left",
        validate="one_to_one",
    )
    training_mask = frame[cohort_col].eq(config.cohort("training"))
    training_scores = frame.loc[training_mask, "vit_score_oof"].to_numpy(float)
    if training_scores.size == 0 or not np.isfinite(training_scores).all():
        raise ValueError("Every training patient must have one finite out-of-fold ViT score.")
    cutoff = float(np.median(training_scores))
    subset = frame.loc[pd.to_numeric(frame[subset_col], errors="coerce").eq(1)].copy()
    if subset.empty:
        raise ValueError("The biological subset contains no patients.")
    subset_scores = subset["vit_score_oof"].to_numpy(float)
    if not np.isfinite(subset_scores).all():
        raise ValueError("Every biological-subset patient requires a finite out-of-fold ViT score.")
    result = subset[[patient_col]].copy()
    result["vit_score"] = subset_scores
    result["vit_cutoff"] = cutoff
    result["vit_score_source"] = "out_of_fold"
    if result["vit_score"].gt(cutoff).nunique() != 2:
        raise ValueError("The locked cutoff must create two groups in the biological subset.")
    return result.sort_values(patient_col).reset_index(drop=True)
