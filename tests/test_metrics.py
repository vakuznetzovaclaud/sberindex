"""Метрики на ручных примерах: тест Диболда–Мариано, MAE / MAPE / WAPE / MASE и метрики детекторов."""
import numpy as np
import pandas as pd
import pytest

from ews import detect_eval as de, evaluate as ev


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_dm_equal_losses_not_significant(rng):
    """Потери различаются только шумом с нулевым средним: разницы нет, p = 1."""
    e1 = rng.gamma(2.0, 1.0, 11)
    noise = rng.normal(0, 0.3, 11)
    e2 = e1 + noise - noise.mean()
    for h in (1, 2, 3):
        assert ev.dm_test(e1, e2, h) == pytest.approx(1.0)


def test_dm_shift_is_significant(rng):
    """Одна модель на всех месяцах хуже на единицу: различие значимо."""
    e2 = rng.gamma(2.0, 1.0, 11)
    e1 = e2 + 1 + rng.normal(0, 0.3, 11)
    for h in (1, 2, 3):
        assert ev.dm_test(e1, e2, h) < 0.05


@pytest.mark.parametrize("h", [1, 2, 3])
def test_dm_symmetric(h, rng):
    e1, e2 = rng.gamma(2.0, 1.0, 11), rng.gamma(2.0, 1.2, 11)
    assert ev.dm_test(e1, e2, h) == pytest.approx(ev.dm_test(e2, e1, h))


def test_dm_negative_variance_falls_back_to_h1():
    """Разность потерь с сильной отрицательной автокорреляцией: оценка дисперсии с лагом 1 отрицательна, и тест для
    h = 2 повторяется с h = 1, как forecast::dm.test в R."""
    d = 0.5 + np.array([1.0, -1.0] * 5 + [1.0])
    n, dc = len(d), d - d.mean()
    assert np.sum(dc ** 2) / n + 2 * np.sum(dc[1:] * dc[:-1]) / n < 0      # условие отката выполнено
    e2 = np.full(n, 2.0)
    p1, p2 = ev.dm_test(e2 + d, e2, 1), ev.dm_test(e2 + d, e2, 2)
    assert np.isfinite(p2) and p2 == p1


def test_dm_too_short():
    assert np.isnan(ev.dm_test([1.0, 2.0], [1.5, 2.5], 1))


def test_forecast_metrics_by_hand(monkeypatch):
    """Два МО, одна точка прогноза (T0 = 13), горизонт 1. Ряды растут на 10 и 20 в месяц, так что знаменатель MASE —
    10 и 20; ошибки +5 и -10 при факте 240 и 580."""
    t = np.arange(15)
    Y = np.array([[100 + 10 * t, 300 + 20 * t]], float)                   # [категория, МО, месяц]
    fc = pd.DataFrame({"i": [0, 1], "origin": [13, 13], "h": [1, 1], "target": [14, 14], "yhat": [245.0, 570.0]})
    monkeypatch.setattr(ev.pnl, "load", lambda: (Y, np.arange(2), [f"m{k}" for k in t], None))
    monkeypatch.setattr(ev, "load", lambda name, cat=0: fc.copy())
    r = ev.table(["m"], 0).iloc[0]
    assert r["MAE"] == pytest.approx(7.5)
    assert r["MAPE, %"] == pytest.approx(100 * (5 / 240 + 10 / 580) / 2)
    assert r["WAPE, %"] == pytest.approx(100 * 15 / 820)
    assert r["MASE"] == pytest.approx((5 / 10 + 10 / 20) / 2)


def _shocks(rng, n=40, t=12, n_shocked=10):
    """Оценки детектора [МО, месяц] для чистых месяцев — шум ниже 1; шок в МО 0…n_shocked - 1 в месяцы 3…8."""
    tau = np.full(n, -1)
    tau[:n_shocked] = rng.integers(3, 9, n_shocked)
    S = rng.uniform(0, 0.9, (n, t))
    return S, tau, list(range(t))


def test_detector_metrics_perfect(rng):
    """Идеальный детектор (с месяца шока оценка выше любой чистой): полнота 1 при любой доле ложных тревог,
    задержка 0, VUS-PR и F1 с допуском 1."""
    S, tau, T = _shocks(rng)
    for i in np.where(tau >= 0)[0]:
        S[i, tau[i]:] = 1.0
    r = de.evaluate(S, tau, T)
    for fa in de.FA_CURVE:
        k = f"{fa * 100:g}%"
        assert r[f"полнота@{k}"] == 1 and r[f"задержка@{k}"] == 0 and r[f"задержка с пропусками@{k}"] == 0
    assert r["VUS-PR"] == pytest.approx(1.0) and r["F1 с допуском"] == pytest.approx(1.0)
    assert r["NAB-подобная"] > 95


def test_detector_metrics_late_by_one_month(rng):
    """Детектор срабатывает через месяц после шока: полнота 1, задержка 1."""
    S, tau, T = _shocks(rng)
    for i in np.where(tau >= 0)[0]:
        S[i, tau[i] + 1] = 1.0
    r = de.evaluate(S, tau, T)
    assert r["полнота@3%"] == 1 and r["задержка@3%"] == 1 and r["задержка с пропусками@3%"] == 1


def test_detector_metrics_constant(rng):
    """Постоянная оценка ничего не ловит (тревога — только выше порога), а VUS-PR равен доле шоковых МО-месяцев."""
    _, tau, T = _shocks(rng)
    S = np.zeros((len(tau), len(T)))
    r = de.evaluate(S, tau, T)
    assert r["полнота@3%"] == 0 and r["задержка с пропусками@3%"] == 3
    Ta = np.array(T)
    share = []
    for buf in range(3):
        pos = sum(((Ta >= tau[i]) & (Ta <= tau[i] + buf)).sum() for i in np.where(tau >= 0)[0])
        clean = (tau < 0).sum() * len(T) + sum((Ta < tau[i]).sum() for i in np.where(tau >= 0)[0])
        share.append(pos / (pos + clean))
    assert r["VUS-PR"] == pytest.approx(np.mean(share))
