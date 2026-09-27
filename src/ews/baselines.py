"""Бейзлайн задания — Prophet, по отдельной модели на каждый ряд МО, в трёх конфигурациях:
по умолчанию (годовая сезонность включается автоматически только при истории от двух лет — у нас её нет),
как в библиотеке TimeCast (годовая и недельная сезонность включены принудительно) и с подбором настроек по сетке
GRID (2 порядка гармоник × 2 масштаба тренда × 2 масштаба сезонности): в каждой точке выбирается настройка с наименьшей
ошибкой по всем целям, наблюдённым к этой точке (select_online)."""
import logging
import warnings

import numpy as np
import pandas as pd

from . import backtest, panel as pnl

CONFIGS = {
    "prophet_default": {},
    "prophet_timecast": dict(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode="additive",
                             changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, holidays_prior_scale=10.0, interval_width=0.8),
}
GRID = [dict(yearly_seasonality=k, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=cp, seasonality_prior_scale=sp)
        for k in (2, 4) for cp in (0.01, 0.1) for sp in (0.1, 1.0)]


def _fit_predict(series, months, T0, H, kw):
    warnings.filterwarnings("ignore")
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    from prophet import Prophet
    ds = pd.to_datetime([m + "-01" for m in months])
    y = series[:T0 + 1]
    m = Prophet(**kw)
    m.fit(pd.DataFrame({"ds": ds[:T0 + 1], "y": y}))
    fut = pd.DataFrame({"ds": [ds[T0] + pd.DateOffset(months=h) for h in H]})
    p = m.predict(fut)["yhat"].to_numpy()
    return p


def prophet_model(kw, n_jobs=10):
    def f(c):
        from joblib import Parallel, delayed
        idx = range(c.y.shape[0])
        out = Parallel(n_jobs=n_jobs, batch_size=8)(delayed(_fit_predict)(c.y[i], c.months, c.T0, c.H, kw) for i in idx)
        return np.array(out)
    return f


def select_online(members, cat=0, name="prophet_tuned"):
    """Подбор настроек без заглядывания вперёд: в каждой точке прогноза T0 берётся конфигурация с наименьшей MAE по
    целям, уже наблюдённым к T0 (target ≤ T0), из прошлых точек; в первой точке — первая конфигурация сетки."""
    Y, _, _, _ = pnl.load()
    P = {m: pd.read_parquet(backtest.DIR / f"{m}__c{cat}.parquet") for m in members}
    for m, p in P.items():
        p["ae"] = np.abs(p.yhat.to_numpy() - Y[cat][p.i, p.target])
    out, chosen = [], {}
    for T0 in sorted(P[members[0]].origin.unique()):
        past = {m: p[(p.target <= T0)].ae.mean() for m, p in P.items()}
        best = min(members, key=lambda m: past[m]) if all(np.isfinite(v) for v in past.values()) else members[0]
        chosen[int(T0)] = best
        out.append(P[best][P[best].origin == T0].drop(columns="ae"))
    pd.concat(out).to_parquet(backtest.DIR / f"{name}__c{cat}.parquet")
    return chosen
