"""Cox objectives, baseline hazards, and survival-function utilities."""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import logsumexp


def negative_cox_partial_log_likelihood(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Mean negative Cox partial log-likelihood with Breslow handling of ties."""

    log_risk = log_risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1)
    if not (len(log_risk) == len(time) == len(event)):
        raise ValueError("Risk, time, and event tensors must have the same length.")
    if log_risk.numel() == 0:
        raise ValueError("Survival tensors must not be empty.")
    if not torch.isfinite(log_risk).all() or not torch.isfinite(time).all():
        raise ValueError("Risk scores and survival times must be finite.")
    if torch.any(time <= 0):
        raise ValueError("Survival times must be greater than zero.")
    if event.dtype != torch.bool and not torch.all((event == 0) | (event == 1)):
        raise ValueError("Event values must use binary zero-one coding.")
    event = event.bool()
    event_times = torch.unique(time[event], sorted=True)
    if event_times.numel() == 0:
        raise ValueError("At least one observed event is required for the Cox loss.")
    log_likelihood = log_risk.new_zeros(())
    event_count = 0
    for event_time in event_times:
        deaths = event & (time == event_time)
        at_risk = time >= event_time
        deaths_count = int(deaths.sum().item())
        log_likelihood = log_likelihood + log_risk[deaths].sum()
        log_likelihood = log_likelihood - deaths_count * torch.logsumexp(log_risk[at_risk], 0)
        event_count += deaths_count
    return -log_likelihood / event_count


def sampled_risk_set_loss(log_risk: torch.Tensor) -> torch.Tensor:
    """Cox contribution for a sampled risk set with the event subject in position zero."""

    if log_risk.ndim != 1 or log_risk.numel() < 2:
        raise ValueError("A sampled risk set must contain one event and at least one comparator.")
    return -log_risk[0] + torch.logsumexp(log_risk, dim=0)


def breslow_baseline_hazard(
    time: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the training-cohort cumulative baseline hazard."""

    time = np.asarray(time, dtype=float)
    raw_event = np.asarray(event)
    log_risk = np.asarray(log_risk, dtype=float)
    if not (time.shape == raw_event.shape == log_risk.shape) or time.ndim != 1:
        raise ValueError("Time, event, and risk arrays must have identical shapes.")
    if time.size == 0:
        raise ValueError("Survival arrays must not be empty.")
    if not np.isfinite(time).all() or not np.isfinite(log_risk).all():
        raise ValueError("Survival times and risk scores must be finite.")
    if np.any(time <= 0):
        raise ValueError("Survival times must be greater than zero.")
    try:
        numeric_event = raw_event.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Event values must use binary zero-one coding.") from error
    if not np.isfinite(numeric_event).all() or not np.isin(numeric_event, [0.0, 1.0]).all():
        raise ValueError("Event values must use binary zero-one coding.")
    event = numeric_event.astype(bool)
    event_times = np.unique(time[event])
    if event_times.size == 0:
        raise ValueError("At least one observed event is required.")
    increments = []
    for event_time in event_times:
        deaths = np.count_nonzero(event & (time == event_time))
        log_denominator = float(logsumexp(log_risk[time >= event_time]))
        increments.append(float(np.exp(np.log(deaths) - log_denominator)))
    cumulative = np.cumsum(np.asarray(increments, dtype=float))
    if not np.isfinite(cumulative).all():
        raise FloatingPointError("The Breslow cumulative baseline hazard is not finite.")
    return event_times, cumulative


def cumulative_hazard_at(
    event_times: np.ndarray,
    cumulative_hazard: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """Evaluate a right-continuous cumulative baseline hazard at fixed horizons."""

    event_times = np.asarray(event_times, dtype=float)
    cumulative_hazard = np.asarray(cumulative_hazard, dtype=float)
    horizons = np.asarray(horizons, dtype=float)
    if event_times.ndim != 1 or cumulative_hazard.ndim != 1:
        raise ValueError("Event times and cumulative hazard must be one-dimensional.")
    if event_times.shape != cumulative_hazard.shape:
        raise ValueError("Event times and cumulative hazard must have identical lengths.")
    if not (
        np.isfinite(event_times).all()
        and np.isfinite(cumulative_hazard).all()
        and np.isfinite(horizons).all()
    ):
        raise ValueError("Hazard inputs and prediction horizons must be finite.")
    if np.any(np.diff(event_times) <= 0) or np.any(np.diff(cumulative_hazard) < 0):
        raise ValueError(
            "Event times and cumulative hazard must be strictly ordered and monotonic."
        )
    if np.any(horizons < 0):
        raise ValueError("Prediction horizons must be nonnegative.")
    indices = np.searchsorted(event_times, horizons, side="right") - 1
    values = np.zeros_like(horizons, dtype=float)
    available = indices >= 0
    values[available] = cumulative_hazard[indices[available]]
    return values


def predict_survival_probabilities(
    log_risk: np.ndarray,
    event_times: np.ndarray,
    cumulative_hazard: np.ndarray,
    horizons: np.ndarray,
) -> np.ndarray:
    """Convert neural log-risk scores into individualized survival probabilities."""

    log_risk = np.asarray(log_risk, dtype=float).reshape(-1)
    if not np.isfinite(log_risk).all():
        raise ValueError("Risk scores must be finite.")
    baseline = cumulative_hazard_at(event_times, cumulative_hazard, horizons)
    result = np.ones((len(log_risk), len(baseline)), dtype=float)
    positive = baseline > 0
    if positive.any():
        log_cumulative_hazard = log_risk[:, None] + np.log(baseline[None, positive])
        cumulative = np.exp(np.clip(log_cumulative_hazard, -745.0, 709.0))
        result[:, positive] = np.exp(-cumulative)
    return result


def structured_survival(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Build the structured survival array required by scikit-survival."""

    raw_event = np.asarray(event)
    time = np.asarray(time, dtype=float)
    if raw_event.shape != time.shape:
        raise ValueError("Event and time arrays must have identical shapes.")
    if raw_event.ndim != 1 or raw_event.size == 0:
        raise ValueError("Event and time arrays must be nonempty and one-dimensional.")
    try:
        numeric_event = raw_event.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Event values must use binary zero-one coding.") from error
    if not np.isfinite(numeric_event).all() or not np.isin(numeric_event, [0.0, 1.0]).all():
        raise ValueError("Event values must use binary zero-one coding.")
    if not np.isfinite(time).all() or np.any(time <= 0):
        raise ValueError("Survival times must be finite and greater than zero.")
    event = numeric_event.astype(bool)
    return np.array(list(zip(event, time, strict=True)), dtype=[("event", "?"), ("time", "<f8")])
