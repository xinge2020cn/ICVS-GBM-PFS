"""IBSI-aligned feature extraction and leakage-safe radiomics model fitting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis

from .config import StudyConfig
from .survival import structured_survival
from .training import survival_strata


def _radiomic_feature_name(modality: str, name: str) -> str:
    """Name modality-independent shape features once and other features by sequence."""

    if name.startswith("original_shape_"):
        return f"shape__{name}"
    return f"{modality}__{name}"


def extract_radiomics_features(
    frame: pd.DataFrame,
    config: StudyConfig,
    parameter_file: str | Path,
) -> pd.DataFrame:
    """Extract configured 3D features from the combined tumor-peritumoral VOI."""

    try:
        from radiomics import featureextractor
    except ImportError as error:
        raise RuntimeError(
            "Radiomics dependencies are unavailable. Install the radiomics optional dependency."
        ) from error
    patient_col = config.column("patient_id")
    path_columns = [
        "preprocessed_t1_path",
        "preprocessed_t2_path",
        "preprocessed_flair_path",
        "preprocessed_ce_t1_path",
        "voi_path",
    ]
    missing = [column for column in [patient_col, *path_columns] if column not in frame]
    if missing:
        raise ValueError(f"Manifest is missing radiomics columns: {', '.join(missing)}")
    extractor = featureextractor.RadiomicsFeatureExtractor(str(Path(parameter_file).resolve()))
    rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        values: dict[str, object] = {patient_col: str(row[patient_col])}
        for modality in ("t1", "t2", "flair", "ce_t1"):
            features = extractor.execute(
                str(row[f"preprocessed_{modality}_path"]),
                str(row["voi_path"]),
                label=1,
            )
            for name, value in features.items():
                if name.startswith("diagnostics_"):
                    continue
                scalar = float(np.asarray(value).reshape(-1)[0])
                output_name = _radiomic_feature_name(modality, name)
                if output_name in values and not np.isclose(float(values[output_name]), scalar):
                    raise ValueError(
                        f"Modality-independent shape feature changed across sequences: {name}"
                    )
                values[output_name] = scalar
        rows.append(values)
    result = pd.DataFrame(rows)
    feature_columns = [column for column in result if column != patient_col]
    finite = np.isfinite(result[feature_columns].to_numpy(float))
    if not finite.all():
        bad_columns = result[feature_columns].columns[~finite.all(axis=0)].tolist()
        raise ValueError(f"Nonfinite radiomic features were extracted: {', '.join(bad_columns)}")
    return result


def _univariable_screen(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    time_column: str,
    event_column: str,
) -> pd.DataFrame:
    rows = []
    for feature in feature_columns:
        values = frame[feature].to_numpy(float)
        if not np.isfinite(values).all() or np.std(values) <= 1e-12:
            continue
        model_frame = frame[[time_column, event_column, feature]].copy()
        model_frame[feature] = (values - values.mean()) / values.std(ddof=0)
        try:
            model = CoxPHFitter(penalizer=1e-8)
            model.fit(model_frame, duration_col=time_column, event_col=event_column)
        except Exception:
            continue
        row = model.summary.loc[feature]
        rows.append(
            {
                "feature": feature,
                "coefficient": float(row["coef"]),
                "hazard_ratio": float(row["exp(coef)"]),
                "p_value": float(row["p"]),
            }
        )
    if not rows:
        raise RuntimeError("No radiomic feature could be fitted in univariable Cox models.")
    return pd.DataFrame(rows).sort_values(["p_value", "feature"]).reset_index(drop=True)


def _negative_partial_log_likelihood(
    time: np.ndarray,
    event: np.ndarray,
    score: np.ndarray,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    score = np.asarray(score, dtype=float)
    if not (time.shape == event.shape == score.shape) or time.ndim != 1:
        raise ValueError("Time, event, and score arrays must be aligned and one-dimensional.")
    if not np.isfinite(time).all() or not np.isfinite(score).all() or np.any(time <= 0):
        raise ValueError("Survival times and scores must be finite, with positive times.")
    total = 0.0
    events = 0
    for event_time in np.unique(time[event]):
        deaths = event & (time == event_time)
        at_risk = time >= event_time
        deaths_count = int(deaths.sum())
        maximum = float(score[at_risk].max())
        log_denominator = maximum + np.log(np.exp(score[at_risk] - maximum).sum())
        total -= float(score[deaths].sum()) - deaths_count * log_denominator
        events += deaths_count
    if events == 0:
        raise ValueError("A validation fold must contain at least one observed event.")
    return total / events


def _fit_lasso_path(features: np.ndarray, outcome: np.ndarray) -> CoxnetSurvivalAnalysis:
    """Fit one complete LASSO-Cox path without importing a penalty grid from another fold."""

    return CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        n_alphas=100,
        alpha_min_ratio=0.01,
        normalize=False,
        max_iter=200000,
        tol=1e-8,
    ).fit(features, outcome)


def _relative_penalty_path(model: CoxnetSurvivalAnalysis) -> np.ndarray:
    """Return the fold-comparable penalty fraction along an automatically fitted path."""

    alphas = np.asarray(model.alphas_, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0 or not np.isfinite(alphas).all():
        raise RuntimeError("LASSO-Cox returned an invalid penalty path.")
    if alphas[0] <= 0 or np.any(alphas <= 0) or np.any(np.diff(alphas) >= 0):
        raise RuntimeError("LASSO-Cox penalties must be positive and strictly decreasing.")
    return alphas / alphas[0]


def fit_radiomics_model(
    manifest: pd.DataFrame,
    features: pd.DataFrame,
    config: StudyConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Fit the screened LASSO-Cox radiomics model and score all locked cohorts."""

    patient_col = config.column("patient_id")
    cohort_col = config.column("cohort")
    time_col = config.column("pfs_time")
    event_col = config.column("pfs_event")
    if features[patient_col].duplicated().any():
        raise ValueError("Radiomics features contain duplicate patient identifiers.")
    manifest_ids = set(manifest[patient_col].astype(str))
    feature_ids = set(features[patient_col].astype(str))
    if manifest_ids != feature_ids:
        raise ValueError("Radiomics features must contain exactly one row for every patient.")
    merged = manifest.merge(features, on=patient_col, how="left", validate="one_to_one")
    feature_columns = [column for column in features if column != patient_col]
    if not feature_columns:
        raise ValueError("Radiomics feature table contains no feature columns.")
    if merged[feature_columns].isna().any().any():
        raise ValueError("Radiomics features are missing for one or more manifest patients.")
    training = merged.loc[merged[cohort_col].eq(config.cohort("training"))].reset_index(drop=True)
    screen = _univariable_screen(
        training,
        feature_columns,
        time_column=time_col,
        event_column=event_col,
    )
    threshold = float(config.section("radiomics")["univariable_p_threshold"])
    candidates = screen.loc[screen["p_value"] < threshold, "feature"].tolist()
    if not candidates:
        raise RuntimeError("No radiomic features passed the prespecified univariable threshold.")
    full_scaler = StandardScaler().fit(training[candidates])
    full_x = full_scaler.transform(training[candidates])
    training_y = structured_survival(training[event_col], training[time_col])
    path_model = _fit_lasso_path(full_x, training_y)
    alphas = np.asarray(path_model.alphas_, dtype=float)
    relative_penalties = _relative_penalty_path(path_model)
    folds = int(config.section("radiomics")["cross_validation_folds"])
    if folds < 2 or len(training) < folds:
        raise ValueError(
            "Radiomics cross-validation requires at least two folds and one patient per fold."
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    strata = survival_strata(training, config, minimum_count=folds)
    fold_loss_rows: list[np.ndarray] = []
    fold_alpha_paths: list[np.ndarray] = []
    fold_penalty_paths: list[np.ndarray] = []
    fold_screen_rows: list[pd.DataFrame] = []
    fold_assignments: list[dict[str, object]] = []
    for fold, (fit_indices, held_out_indices) in enumerate(splitter.split(training, strata)):
        fit_frame = training.iloc[fit_indices]
        held_out_frame = training.iloc[held_out_indices]
        fold_screen = _univariable_screen(
            fit_frame,
            feature_columns,
            time_column=time_col,
            event_column=event_col,
        )
        fold_screen.insert(0, "fold", fold)
        fold_screen["passed_threshold"] = fold_screen["p_value"] < threshold
        fold_screen_rows.append(fold_screen)
        fold_candidates = fold_screen.loc[fold_screen["passed_threshold"], "feature"].tolist()
        if not fold_candidates:
            raise RuntimeError(
                f"No radiomic features passed the prespecified threshold in cross-validation "
                f"fold {fold}."
            )
        fold_scaler = StandardScaler().fit(fit_frame[fold_candidates])
        fit_x = fold_scaler.transform(fit_frame[fold_candidates])
        held_out_x = fold_scaler.transform(held_out_frame[fold_candidates])
        fit_y = structured_survival(
            fit_frame[event_col], fit_frame[time_col]
        )
        held_out_time = held_out_frame[time_col].to_numpy(float)
        held_out_event = held_out_frame[event_col].to_numpy(bool)
        fold_path = _fit_lasso_path(fit_x, fit_y)
        fold_alpha_paths.append(np.asarray(fold_path.alphas_, dtype=float))
        fold_penalty_paths.append(_relative_penalty_path(fold_path))
        losses = np.full(len(fold_path.alphas_), np.nan, dtype=float)
        for alpha_index, alpha in enumerate(fold_path.alphas_):
            try:
                score = fold_path.predict(held_out_x, alpha=float(alpha))
                losses[alpha_index] = _negative_partial_log_likelihood(
                    held_out_time, held_out_event, score
                )
            except Exception:
                continue
        fold_loss_rows.append(losses)
        fold_assignments.extend(
            {patient_col: str(patient_id), "cross_validation_fold": fold}
            for patient_id in held_out_frame[patient_col]
        )
    common_path_length = min(
        len(relative_penalties),
        *(len(path) for path in fold_penalty_paths),
        *(len(losses) for losses in fold_loss_rows),
    )
    if common_path_length == 0:
        raise RuntimeError("LASSO-Cox cross-validation returned an empty common penalty path.")
    relative_penalties = relative_penalties[:common_path_length]
    alphas = alphas[:common_path_length]
    for fold_path in fold_penalty_paths:
        if not np.allclose(
            fold_path[:common_path_length], relative_penalties, rtol=1e-7, atol=1e-12
        ):
            raise RuntimeError("Cross-validation folds returned incompatible penalty paths.")
    fold_losses = np.vstack([losses[:common_path_length] for losses in fold_loss_rows])
    valid = np.isfinite(fold_losses).all(axis=0)
    if not valid.any():
        raise RuntimeError("LASSO-Cox cross-validation failed for every penalty value.")
    mean_loss = np.nanmean(fold_losses, axis=0)
    standard_error = np.nanstd(fold_losses, axis=0, ddof=1) / np.sqrt(folds)
    best_index = int(np.nanargmin(np.where(valid, mean_loss, np.nan)))
    if bool(config.section("radiomics")["use_one_standard_error_rule"]):
        eligible = np.flatnonzero(
            valid & (mean_loss <= mean_loss[best_index] + standard_error[best_index])
        )
        selected_alpha_index = int(eligible[0])
    else:
        selected_alpha_index = best_index
    selected_alpha = float(alphas[selected_alpha_index])
    selected_relative_penalty = float(relative_penalties[selected_alpha_index])
    path_coefficients = path_model.coef_[:, selected_alpha_index]
    nonzero = np.flatnonzero(np.abs(path_coefficients) > 1e-10)
    if nonzero.size == 0:
        raise RuntimeError(
            "The selected LASSO penalty retained no radiomic features; no feature was substituted."
        )
    selected_features = [candidates[index] for index in nonzero]
    final_scaler = StandardScaler().fit(training[selected_features])
    final_x = final_scaler.transform(training[selected_features])
    final_model = CoxPHSurvivalAnalysis(alpha=1e-8, ties="breslow").fit(final_x, training_y)
    all_x = final_scaler.transform(merged[selected_features])
    scores = final_model.predict(all_x)
    horizons = np.asarray(config.section("icvs")["horizons_months"], dtype=float)
    survival_functions = final_model.predict_survival_function(all_x)
    survival = np.column_stack(
        [[float(function(horizon)) for function in survival_functions] for horizon in horizons]
    )
    training_mask = merged[cohort_col].eq(config.cohort("training")).to_numpy()
    cutoff = float(np.median(scores[training_mask]))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_table = merged[[patient_col, cohort_col]].copy()
    score_table["radiomics_score"] = scores
    score_table["radiomics_risk_group"] = np.where(scores > cutoff, "high", "low")
    score_table["radiomics_training_cutoff"] = cutoff
    for horizon_index, horizon in enumerate(horizons):
        score_table[f"radiomics_pfs_{int(horizon)}m"] = survival[:, horizon_index]
    score_table.to_csv(output / "radiomics_scores.csv", index=False)
    coefficients = pd.DataFrame(
        {
            "feature": selected_features,
            "coefficient": final_model.coef_,
            "training_mean": final_scaler.mean_,
            "training_scale": final_scaler.scale_,
        }
    )
    coefficients.to_csv(output / "radiomics_selected_features.csv", index=False)
    screen.to_csv(output / "radiomics_univariable_screen.csv", index=False)
    pd.concat(fold_screen_rows, ignore_index=True).to_csv(
        output / "radiomics_fold_screening.csv", index=False
    )
    pd.DataFrame(fold_assignments).sort_values("cross_validation_fold").to_csv(
        output / "radiomics_cross_validation_assignments.csv", index=False
    )
    penalty_rows = []
    for fold, (fold_alphas, fold_relative, fold_loss) in enumerate(
        zip(fold_alpha_paths, fold_penalty_paths, fold_loss_rows, strict=True)
    ):
        penalty_rows.extend(
            {
                "fold": fold,
                "alpha": float(alpha),
                "relative_penalty": float(relative),
                "negative_partial_log_likelihood": float(loss),
                "included_in_common_path": index < common_path_length,
            }
            for index, (alpha, relative, loss) in enumerate(
                zip(fold_alphas, fold_relative, fold_loss, strict=True)
            )
        )
    pd.DataFrame(penalty_rows).to_csv(output / "radiomics_fold_penalty_losses.csv", index=False)
    cross_validation = pd.DataFrame(
        {
            "alpha": alphas,
            "relative_penalty": relative_penalties,
            "mean_negative_partial_log_likelihood": mean_loss,
            "standard_error": standard_error,
            "selected": np.arange(len(alphas)) == selected_alpha_index,
        }
    )
    cross_validation.to_csv(output / "radiomics_penalty_selection.csv", index=False)
    joblib.dump(
        {
            "model": final_model,
            "scaler": final_scaler,
            "features": selected_features,
            "selected_alpha": selected_alpha,
            "selected_relative_penalty": selected_relative_penalty,
            "training_cutoff": cutoff,
            "horizons_months": horizons,
            "training_patient_ids": training[patient_col].astype(str).tolist(),
        },
        output / "radiomics_model.joblib",
    )
    (output / "radiomics_model_metadata.json").write_text(
        json.dumps(
            {
                "univariable_p_threshold": threshold,
                "candidate_features": len(candidates),
                "selected_features": len(selected_features),
                "selected_alpha": selected_alpha,
                "selected_relative_penalty": selected_relative_penalty,
                "training_cutoff": cutoff,
                "cross_validation_folds": folds,
                "univariable_screening_repeated_within_each_fold": True,
                "fold_candidate_features": [
                    int(frame["passed_threshold"].sum()) for frame in fold_screen_rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return score_table
