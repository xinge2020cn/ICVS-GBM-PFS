import numpy as np
import pytest

torch = pytest.importorskip("torch")

from icvs_gbm_pfs.survival import (  # noqa: E402
    breslow_baseline_hazard,
    negative_cox_partial_log_likelihood,
    predict_survival_probabilities,
    sampled_risk_set_loss,
)


def test_breslow_cox_loss_for_one_event() -> None:
    loss = negative_cox_partial_log_likelihood(
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
    )
    assert loss.item() == pytest.approx(np.log(2.0))


def test_sampled_risk_set_loss_matches_single_event_contribution() -> None:
    loss = sampled_risk_set_loss(torch.tensor([0.0, 0.0, 0.0, 0.0]))
    assert loss.item() == pytest.approx(np.log(4.0))


def test_breslow_baseline_and_survival_probability() -> None:
    times, cumulative = breslow_baseline_hazard(
        np.array([1.0, 2.0]),
        np.array([1, 0]),
        np.array([0.0, 0.0]),
    )
    assert times.tolist() == [1.0]
    assert cumulative.tolist() == pytest.approx([0.5])
    survival = predict_survival_probabilities(
        np.array([0.0]), times, cumulative, np.array([0.5, 1.0, 2.0])
    )
    assert survival[0, 0] == pytest.approx(1.0)
    assert survival[0, 1] == pytest.approx(np.exp(-0.5))
    assert survival[0, 2] == pytest.approx(np.exp(-0.5))


def test_structured_survival_rejects_fractional_event_values() -> None:
    from icvs_gbm_pfs.survival import structured_survival

    with pytest.raises(ValueError, match="binary"):
        structured_survival(np.array([1.0, 0.5]), np.array([1.0, 2.0]))


def test_breslow_baseline_is_stable_for_large_log_risk() -> None:
    times, cumulative = breslow_baseline_hazard(
        np.array([1.0, 2.0]),
        np.array([1, 0]),
        np.array([700.0, 700.0]),
    )
    assert times.tolist() == [1.0]
    assert np.isfinite(cumulative).all()
    assert cumulative[0] > 0.0


def test_breslow_does_not_merge_close_distinct_event_times() -> None:
    times, cumulative = breslow_baseline_hazard(
        np.array([100.0, 100.0005]),
        np.array([1, 1]),
        np.array([0.0, 0.0]),
    )
    assert times.tolist() == [100.0, 100.0005]
    assert cumulative.tolist() == pytest.approx([0.5, 1.5])
