"""Metric semantics, especially skill-score sign conventions."""

import pandas as pd
import pytest

from src.metrics import score_all


def series(*vals):
    return pd.Series(list(vals))


def test_skill_is_zero_when_model_equals_persistence():
    """Copying today must score exactly 0 skill - not positive, not negative."""
    today = series("Good", "Good", "Poor", "Severe")
    truth = series("Good", "Poor", "Poor", "Good")
    m = score_all(truth, today, today_category=today)
    assert m["skill_score"] == pytest.approx(0.0)


def test_skill_is_one_for_a_perfect_forecast():
    today = series("Good", "Good", "Poor")
    truth = series("Good", "Poor", "Good")
    m = score_all(truth, truth, today_category=today)
    assert m["skill_score"] == pytest.approx(1.0)


def test_skill_is_negative_when_worse_than_doing_nothing():
    today = series("Good", "Good", "Good", "Good")
    truth = series("Good", "Good", "Good", "Poor")  # persistence gets 3/4
    bad = series("Severe", "Severe", "Severe", "Severe")  # model gets 0/4
    m = score_all(truth, bad, today_category=today)
    assert m["skill_score"] < 0


def test_persistence_cannot_predict_any_change():
    """The structural blind spot: 1.0 on stable days, 0.0 on changes."""
    today = series("Good", "Good", "Poor", "Poor")
    truth = series("Good", "Severe", "Poor", "Good")
    m = score_all(truth, today, today_category=today)
    assert m["accuracy_stable_days"] == pytest.approx(1.0)
    assert m["accuracy_change_days"] == pytest.approx(0.0)
    assert m["recall_deterioration"] == pytest.approx(0.0)


def test_deterioration_recall_counts_only_worsening_days():
    today = series("Good", "Poor", "Good")
    truth = series("Severe", "Good", "Good")  # only row 0 worsens
    pred = series("Severe", "Good", "Good")
    m = score_all(truth, pred, today_category=today)
    assert m["n_deteriorations"] == 1
    assert m["recall_deterioration"] == pytest.approx(1.0)


def test_bad_air_recall_uses_poor_as_the_threshold():
    truth = series("Poor", "Very Poor", "Moderate", "Good")
    pred = series("Poor", "Moderate", "Moderate", "Good")  # misses one bad day
    m = score_all(truth, pred)
    assert m["bad_air_n_actual"] == 2
    assert m["bad_air_recall"] == pytest.approx(0.5)


def test_per_category_recall_reported_for_present_classes_only():
    truth = series("Good", "Good", "Severe")
    pred = series("Good", "Severe", "Severe")
    m = score_all(truth, pred)
    assert m["recall_good"] == pytest.approx(0.5)
    assert m["recall_severe"] == pytest.approx(1.0)
    assert "recall_poor" not in m  # never occurred, so no misleading 0.0
