"""Leakage-safe training and cross-fitting for three-dimensional survival networks."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .config import StudyConfig
from .datasets import SurvivalVolumeDataset
from .models import build_deep_survival_model
from .reproducibility import set_global_seed, write_environment_report
from .survival import (
    breslow_baseline_hazard,
    negative_cox_partial_log_likelihood,
    predict_survival_probabilities,
    sampled_risk_set_loss,
)


@dataclass(frozen=True)
class TrainingResult:
    """State and history returned by one network fit."""

    state_dict: dict[str, torch.Tensor]
    epochs_completed: int
    best_epoch: int
    history: list[dict[str, float]]


class RiskSetSampler:
    """Sample one observed event and comparators still at risk at its event time."""

    def __init__(
        self,
        time: np.ndarray,
        event: np.ndarray,
        *,
        risk_set_size: int,
        seed: int,
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.event = np.asarray(event, dtype=bool)
        self.risk_set_size = int(risk_set_size)
        self.rng = np.random.default_rng(seed)
        self.event_indices = np.flatnonzero(self.event)
        if self.event_indices.size == 0:
            raise ValueError("Deep survival training requires at least one observed event.")
        if self.risk_set_size < 2:
            raise ValueError("Risk-set size must be at least two.")

    def epoch(self) -> list[np.ndarray]:
        event_indices = self.rng.permutation(self.event_indices)
        sampled = []
        for event_index in event_indices:
            candidates = np.flatnonzero(self.time >= self.time[event_index])
            candidates = candidates[candidates != event_index]
            if candidates.size == 0:
                continue
            replace = candidates.size < self.risk_set_size - 1
            comparators = self.rng.choice(
                candidates,
                size=self.risk_set_size - 1,
                replace=replace,
            )
            sampled.append(np.concatenate([[event_index], comparators]))
        if not sampled:
            raise ValueError("No valid sampled risk sets could be constructed.")
        return sampled


def survival_strata(
    frame: pd.DataFrame,
    config: StudyConfig,
    bins: int = 4,
    minimum_count: int = 2,
) -> np.ndarray:
    """Create joint event and observed-time strata for patient-level splitting."""

    event = frame[config.column("pfs_event")].astype(int).to_numpy()
    time = frame[config.column("pfs_time")].astype(float)
    try:
        time_bin = pd.qcut(time, q=bins, labels=False, duplicates="drop").astype(int)
    except ValueError:
        time_bin = pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    strata = np.char.add(event.astype(str), np.char.add("_", time_bin.to_numpy().astype(str)))
    counts = pd.Series(strata).value_counts()
    rare = set(counts[counts < minimum_count].index)
    if rare:
        strata = np.array(
            [
                str(event_value) if value in rare else value
                for value, event_value in zip(strata, event, strict=True)
            ]
        )
    return strata


def _stack_risk_set(
    dataset: SurvivalVolumeDataset,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    images = [dataset[int(index)]["image"] for index in indices]
    return torch.stack(images).to(device, non_blocking=True)


def predict_log_risk(
    model: torch.nn.Module,
    dataset: SurvivalVolumeDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Predict continuous log-risk scores in manifest order."""

    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            image = torch.stack([dataset[index]["image"] for index in range(start, stop)])
            predictions.append(model(image.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(predictions).astype(float)


def _validation_loss(
    model: torch.nn.Module,
    dataset: SurvivalVolumeDataset,
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    predictions = predict_log_risk(model, dataset, device=device, batch_size=batch_size)
    with torch.no_grad():
        loss = negative_cox_partial_log_likelihood(
            torch.from_numpy(predictions),
            torch.from_numpy(frame[config.column("pfs_time")].to_numpy(np.float32)),
            torch.from_numpy(frame[config.column("pfs_event")].to_numpy(bool)),
        )
    return float(loss.item())


def fit_deep_survival_model(
    model: torch.nn.Module,
    training_frame: pd.DataFrame,
    config: StudyConfig,
    *,
    model_name: str,
    device: torch.device,
    cache_dir: str | Path | None,
    epochs: int,
    validation_frame: pd.DataFrame | None = None,
    patience: int | None = None,
    seed_offset: int = 0,
) -> TrainingResult:
    """Fit a network using sampled Cox risk sets and patient-level validation."""

    set_global_seed(config.seed + seed_offset)
    settings = config.section("deep_survival")
    model_settings = settings[model_name]
    optimization = settings["optimization"]
    physical_batch_size = int(model_settings["physical_batch_size"])
    accumulation_steps = int(model_settings["accumulation_steps"])
    training_dataset = SurvivalVolumeDataset(
        training_frame,
        config,
        augment=True,
        cache_dir=cache_dir,
        seed_offset=seed_offset,
    )
    validation_dataset = None
    if validation_frame is not None:
        validation_dataset = SurvivalVolumeDataset(
            validation_frame,
            config,
            augment=False,
            cache_dir=cache_dir,
            seed_offset=seed_offset,
        )
    sampler = RiskSetSampler(
        training_frame[config.column("pfs_time")].to_numpy(float),
        training_frame[config.column("pfs_event")].to_numpy(bool),
        risk_set_size=physical_batch_size,
        seed=config.seed + seed_offset,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    warmup_epochs = int(optimization["warmup_epochs"])
    minimum_learning_rate = float(optimization["minimum_learning_rate"])
    initial_learning_rate = float(optimization["learning_rate"])

    def learning_rate_factor(epoch_index: int) -> float:
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / max(warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs) / max(epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        floor = minimum_learning_rate / initial_learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        cumulative_loss = 0.0
        risk_sets = sampler.epoch()
        for step, indices in enumerate(risk_sets, start=1):
            images = _stack_risk_set(training_dataset, indices, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                risk = model(images)
                loss = sampled_risk_set_loss(risk) / accumulation_steps
            scaler.scale(loss).backward()
            cumulative_loss += float(loss.detach().cpu()) * accumulation_steps
            if step % accumulation_steps == 0 or step == len(risk_sets):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(optimization["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        training_loss = cumulative_loss / len(risk_sets)
        if validation_dataset is not None and validation_frame is not None:
            validation_loss = _validation_loss(
                model,
                validation_dataset,
                validation_frame,
                config,
                device=device,
                batch_size=physical_batch_size,
            )
        else:
            validation_loss = training_loss
        history.append(
            {
                "epoch": float(epoch + 1),
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if patience is not None and stale_epochs >= patience:
            break
    if validation_dataset is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history)
    return TrainingResult(
        state_dict={name: value.detach().cpu() for name, value in best_state.items()},
        epochs_completed=len(history),
        best_epoch=best_epoch,
        history=history,
    )


def select_training_duration(
    training_frame: pd.DataFrame,
    config: StudyConfig,
    *,
    model_name: str,
    device: torch.device,
    cache_dir: str | Path | None,
) -> TrainingResult:
    """Select training duration using the prespecified internal tuning partition."""

    optimization = config.section("deep_survival")["optimization"]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(optimization["tuning_fraction"]),
        random_state=config.seed,
    )
    development_indices, tuning_indices = next(
        splitter.split(training_frame, survival_strata(training_frame, config, minimum_count=2))
    )
    development = training_frame.iloc[development_indices].reset_index(drop=True)
    tuning = training_frame.iloc[tuning_indices].reset_index(drop=True)
    target_shape = config.section("preprocessing")["target_shape_dhw"]
    model = build_deep_survival_model(
        model_name,
        config.section("deep_survival"),
        target_shape_dhw=target_shape,
    )
    return fit_deep_survival_model(
        model,
        development,
        config,
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        epochs=int(optimization["maximum_tuning_epochs"]),
        validation_frame=tuning,
        patience=int(optimization["early_stopping_patience"]),
    )


def run_crossfit_and_refit(
    frame: pd.DataFrame,
    config: StudyConfig,
    *,
    model_name: str,
    output_dir: str | Path,
    cache_dir: str | Path | None,
    device_name: str | None = None,
    epochs_override: int | None = None,
) -> pd.DataFrame:
    """Generate training out-of-fold scores, refit once, and score locked cohorts."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_environment_report(output / "environment.json")
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    cohort_col = config.column("cohort")
    patient_col = config.column("patient_id")
    training_mask = frame[cohort_col].eq(config.cohort("training"))
    training = frame.loc[training_mask].reset_index(drop=True)
    if training.empty:
        raise ValueError("The training cohort is empty.")
    settings = config.section("deep_survival")
    epochs = int(epochs_override or settings[model_name]["epochs"])
    folds = int(settings["optimization"]["crossfit_folds"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.seed)
    out_of_fold = np.full(len(training), np.nan, dtype=float)
    target_shape = config.section("preprocessing")["target_shape_dhw"]
    fold_records = []
    fold_assignments = []
    for fold, (fit_indices, held_out_indices) in enumerate(
        splitter.split(training, survival_strata(training, config, minimum_count=folds))
    ):
        model = build_deep_survival_model(
            model_name,
            settings,
            target_shape_dhw=target_shape,
        )
        result = fit_deep_survival_model(
            model,
            training.iloc[fit_indices].reset_index(drop=True),
            config,
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            epochs=epochs,
            seed_offset=fold + 1,
        )
        model.load_state_dict(result.state_dict)
        model.to(device)
        held_out_dataset = SurvivalVolumeDataset(
            training.iloc[held_out_indices].reset_index(drop=True),
            config,
            augment=False,
            cache_dir=cache_dir,
        )
        out_of_fold[held_out_indices] = predict_log_risk(
            model,
            held_out_dataset,
            device=device,
            batch_size=int(settings[model_name]["physical_batch_size"]),
        )
        fold_path = output / f"{model_name}_fold_{fold}.pt"
        torch.save(
            {
                "model_name": model_name,
                "fold": fold,
                "state_dict": result.state_dict,
                "fit_patient_ids": training.iloc[fit_indices][patient_col].astype(str).tolist(),
                "held_out_patient_ids": training.iloc[held_out_indices][patient_col]
                .astype(str)
                .tolist(),
                "epochs": epochs,
            },
            fold_path,
        )
        fold_records.append(
            {
                "fold": fold,
                "fit_patients": len(fit_indices),
                "held_out_patients": len(held_out_indices),
                "epochs": epochs,
            }
        )
        fold_assignments.extend(
            {
                patient_col: str(patient_id),
                "crossfit_fold": fold,
            }
            for patient_id in training.iloc[held_out_indices][patient_col]
        )
    if not np.isfinite(out_of_fold).all():
        raise RuntimeError(
            "Cross-fitting did not produce exactly one score for every training patient."
        )
    final_model = build_deep_survival_model(
        model_name,
        settings,
        target_shape_dhw=target_shape,
    )
    final_result = fit_deep_survival_model(
        final_model,
        training,
        config,
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        epochs=epochs,
        seed_offset=folds + 1,
    )
    final_model.load_state_dict(final_result.state_dict)
    final_model.to(device)
    full_dataset = SurvivalVolumeDataset(frame, config, augment=False, cache_dir=cache_dir)
    final_scores = predict_log_risk(
        final_model,
        full_dataset,
        device=device,
        batch_size=int(settings[model_name]["physical_batch_size"]),
    )
    training_positions = np.flatnonzero(training_mask.to_numpy())
    score_table = frame[[patient_col, cohort_col]].copy()
    score_table[f"{model_name}_score_final"] = final_scores
    score_table[f"{model_name}_score_oof"] = np.nan
    score_table.loc[training_positions, f"{model_name}_score_oof"] = out_of_fold
    event_times, cumulative_hazard = breslow_baseline_hazard(
        training[config.column("pfs_time")].to_numpy(float),
        training[config.column("pfs_event")].to_numpy(bool),
        final_scores[training_positions],
    )
    oof_event_times, oof_cumulative_hazard = breslow_baseline_hazard(
        training[config.column("pfs_time")].to_numpy(float),
        training[config.column("pfs_event")].to_numpy(bool),
        out_of_fold,
    )
    horizons = np.asarray(config.section("icvs")["horizons_months"], dtype=float)
    survival = predict_survival_probabilities(
        final_scores,
        event_times,
        cumulative_hazard,
        horizons,
    )
    survival[training_positions] = predict_survival_probabilities(
        out_of_fold,
        oof_event_times,
        oof_cumulative_hazard,
        horizons,
    )
    for horizon_index, horizon in enumerate(horizons):
        score_table[f"{model_name}_pfs_{int(horizon)}m"] = survival[:, horizon_index]
    score_table.to_csv(output / f"{model_name}_scores.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / f"{model_name}_crossfit_folds.csv", index=False)
    pd.DataFrame(fold_assignments).sort_values("crossfit_fold").to_csv(
        output / f"{model_name}_crossfit_assignments.csv", index=False
    )
    artifact = {
        "model_name": model_name,
        "state_dict": final_result.state_dict,
        "input_shape_dhw": [int(value) for value in target_shape],
        "event_times": event_times,
        "baseline_cumulative_hazard": cumulative_hazard,
        "horizons_months": horizons,
        "epochs": epochs,
        "training_patient_ids": training[patient_col].astype(str).tolist(),
        "configuration": settings,
    }
    torch.save(artifact, output / f"{model_name}_locked_model.pt")
    (output / f"{model_name}_training_history.json").write_text(
        json.dumps(final_result.history, indent=2) + "\n", encoding="utf-8"
    )
    return score_table
