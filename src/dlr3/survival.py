"""Cox objectives, baseline hazards, and survival-function utilities."""

from __future__ import annotations

import numpy as np
import torch


def negative_cox_partial_log_likelihood(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Mean negative Cox partial log-likelihood with Breslow handling of ties."""

    log_risk = log_risk.reshape(-1)
    time = time.reshape(-1)
    event = event.reshape(-1).bool()
    if not (len(log_risk) == len(time) == len(event)):
        raise ValueError("Risk, time, and event tensors must have the same length.")
    event_times = torch.unique(time[event], sorted=True)
    if event_times.numel() == 0:
        raise ValueError("At least one observed event is required for the Cox loss.")
    log_likelihood = log_risk.new_zeros(())
    event_count = 0
    for event_time in event_times:
        deaths = event & torch.isclose(time, event_time)
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
    event = np.asarray(event, dtype=bool)
    log_risk = np.asarray(log_risk, dtype=float)
    if not (time.shape == event.shape == log_risk.shape):
        raise ValueError("Time, event, and risk arrays must have identical shapes.")
    event_times = np.unique(time[event])
    if event_times.size == 0:
        raise ValueError("At least one observed event is required.")
    centered_risk = log_risk - np.max(log_risk)
    relative_risk = np.exp(centered_risk)
    increments = []
    for event_time in event_times:
        deaths = np.count_nonzero(event & np.isclose(time, event_time))
        denominator = relative_risk[time >= event_time].sum()
        if denominator <= 0:
            raise FloatingPointError("The Breslow risk-set denominator is not positive.")
        increments.append(deaths / denominator)
    cumulative = np.cumsum(np.asarray(increments, dtype=float))
    cumulative *= np.exp(-np.max(log_risk))
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
    baseline = cumulative_hazard_at(event_times, cumulative_hazard, horizons)
    return np.exp(-np.exp(log_risk[:, None]) * baseline[None, :])


def structured_survival(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Build the structured survival array required by scikit-survival."""

    event = np.asarray(event).astype(bool)
    time = np.asarray(time, dtype=float)
    return np.array(list(zip(event, time, strict=True)), dtype=[("event", "?"), ("time", "<f8")])
