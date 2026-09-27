"""Фундаментальные модели временных рядов (веса с открытыми лицензиями Apache-2.0) на отклонениях МО от общего фактора.
Сырой ряд из 13–23 точек содержит один годовой цикл — FM не видят годовую сезонность и проваливают декабрь; поэтому
общий фактор прогнозируется отдельно (factor.py), а FM прогнозирует только отклонение МО."""
import tempfile

import numpy as np
import pandas as pd

from . import factor as fct
from .models import Ctx, _decompose, factor_forecast, sa_deviation
from .paths import config

_CACHE = {}


def _weights(name):
    """Репозиторий весов на Hugging Face и закреплённая ревизия (configs/base.yaml, foundation)."""
    w = config()["foundation"][name]
    return w["repo"], w["revision"]


def _chronos2():
    if "chronos2" not in _CACHE:
        from chronos import BaseChronosPipeline
        repo, rev = _weights("chronos2")
        _CACHE["chronos2"] = BaseChronosPipeline.from_pretrained(repo, revision=rev, device_map="cpu")
    return _CACHE["chronos2"]


def _timesfm25():
    if "timesfm25" not in _CACHE:
        import timesfm
        repo, rev = _weights("timesfm25")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo, revision=rev)
        m.compile(timesfm.ForecastConfig(max_context=64, max_horizon=16, normalize_inputs=True, use_continuous_quantile_head=True))
        _CACHE["timesfm25"] = m
    return _CACHE["timesfm25"]


def _tirex2():
    if "tirex2" not in _CACHE:
        from tirex2 import load_model
        repo, rev = _weights("tirex2")
        _CACHE["tirex2"] = load_model(repo, device="cpu", hf_kwargs={"revision": rev})
    return _CACHE["tirex2"]


def _q50(q):
    """Медиана из ответа predict_quantiles Chronos-2 (запрошен один квантиль, 0,5): по каждому ряду — вектор длины h."""
    return np.stack([qq[..., 0].numpy().ravel() for qq in q])


def _median_forecast(name, ctx, h):
    import torch
    if name == "chronos2":
        q, _ = _chronos2().predict_quantiles(inputs=torch.tensor(ctx[:, None, :], dtype=torch.float32), prediction_length=h, quantile_levels=[0.5])
        return _q50(q)
    if name == "timesfm25":
        pt, _ = _timesfm25().forecast(horizon=h, inputs=[r for r in ctx])
        return np.asarray(pt)[:, :h]
    if name == "tirex2":
        from tirex2 import TimeseriesType
        tss = [TimeseriesType(target=torch.tensor(r, dtype=torch.float32).unsqueeze(0), past_covariates=None, future_covariates=None) for r in ctx]
        return np.stack([np.asarray(o)[0, 4, :h] for o in _tirex2().forecast(tss, prediction_length=h, output_type="numpy")])   # 4 — квантиль 0,5 из 0,1…0,9
    raise ValueError(name)


def fm_on_deviation(name):
    def f(c: Ctx):
        F, U = _decompose(c)
        uh = _median_forecast(name, U[:, :c.T0 + 1], max(c.H))
        fh = factor_forecast(c)
        return np.stack([np.exp(fh[j] + uh[:, h - 1]) for j, h in enumerate(c.H)], 1)
    return f


def _chronos2_df(ctx, h, groups, covs=(), cross=False, batch=100):
    """Chronos-2 через predict_df: ряды упорядочены по группе (коду региона) и режутся на батчи по batch рядов подряд;
    при cross=True ряды одного батча видят друг друга (cross-learning), так что МО соседних по коду регионов
    прогнозируются вместе. covs — прошлые ковариаты того же МО (их будущие значения неизвестны)."""
    N, L = ctx.shape
    order = np.argsort(groups, kind="stable")
    ts = pd.date_range("2000-01-01", periods=L + h, freq="MS")
    df = pd.DataFrame({"item_id": np.repeat(np.arange(N), L), "timestamp": np.tile(ts[:L], N), "target": ctx[order].ravel(),
                       **{f"cov{k}": cv[order].ravel() for k, cv in enumerate(covs)}})
    pr = _chronos2().predict_df(df, prediction_length=h, quantile_levels=[0.5], cross_learning=cross, batch_size=batch)
    pv = pr.pivot_table(index="item_id", columns="timestamp", values="predictions").reindex(index=range(N), columns=ts[L:L + h]).to_numpy()
    out = np.empty_like(pv)
    out[order] = pv
    return out


def fm_on_seasonally_adjusted(name, lam=0.7, cross=False):
    """FM на отклонении МО, из которого вычтен собственный сезонный профиль МО за первый год панели (сжатый к нулю на lam);
    профиль возвращается к прогнозу. Годовой цикл FM по 13–23 точкам не выучит — отдаём его готовым, FM прогнозирует
    уровень и тренд отклонения. Профиль берётся из 2023 г., известного в любой точке прогноза 2024 г.
    cross=True — Chronos-2 совместно по батчам МО, упорядоченных по коду региона (_chronos2_df)."""
    def f(c: Ctx):
        F, U = _decompose(c)
        usa, s = sa_deviation(U, c.T0, lam)
        if cross:
            uh = _chronos2_df(usa, max(c.H), c.meta["region_code"].astype(int).to_numpy(), cross=True)
        else:
            uh = _median_forecast(name, usa, max(c.H))
        fh = factor_forecast(c)
        return np.stack([np.exp(fh[j] + uh[:, h - 1] + s[:, (c.T0 + h) % 12]) for j, h in enumerate(c.H)], 1)
    return f


def chronos2_sa_covariates(lam=0.7, cross=False):
    """Chronos-2 с ковариатами — возможность, которой нет у TimesFM-2.5 и TiRex-2 в нашей постановке: к сезонно
    скорректированному отклонению МО по категории добавляются такие же ряды пяти остальных категорий того же МО
    (многомерный вход, только прошлое). cross — совместный прогноз батча (≈100 МО × 6 рядов, соседних по коду региона)."""
    def f(c: Ctx):
        sa = []
        for k in range(c.ya.shape[0]):
            LY = np.log(c.ya[k])
            sa.append(sa_deviation(LY - fct.factor(LY), c.T0, lam))
        usa, s = sa[c.cat]
        covs = [u for k, (u, _) in enumerate(sa) if k != c.cat]
        uh = _chronos2_df(usa, max(c.H), c.meta["region_code"].astype(int).to_numpy(), covs, cross, 100 * (1 + len(covs)))
        fh = factor_forecast(c)
        return np.stack([np.exp(fh[j] + uh[:, h - 1] + s[:, (c.T0 + h) % 12]) for j, h in enumerate(c.H)], 1)
    return f


def chronos2_finetuned_sa(lam=0.7, steps=300, lr=1e-4):
    """Chronos-2, дообученный целиком (все веса) в каждой точке прогноза на сезонно скорректированных отклонениях всех
    МО до T0 включительно."""
    def f(c: Ctx):
        import torch
        F, U = _decompose(c)
        usa, s = sa_deviation(U, c.T0, lam)
        h = max(c.H)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            ft = _chronos2().fit(inputs=[r.astype(np.float32) for r in usa], prediction_length=h, finetune_mode="full",
                                 learning_rate=lr, num_steps=steps, batch_size=64, output_dir=tmp, min_past=6,
                                 remove_printer_callback=True, disable_tqdm=True)
            q, _ = ft.predict_quantiles(inputs=torch.tensor(usa[:, None, :], dtype=torch.float32), prediction_length=h,
                                        quantile_levels=[0.5])
        uh = _q50(q)
        fh = factor_forecast(c)
        return np.stack([np.exp(fh[j] + uh[:, hh - 1] + s[:, (c.T0 + hh) % 12]) for j, hh in enumerate(c.H)], 1)
    return f


def fm_on_raw(name):
    """FM на логарифме сырого ряда МО, без панельной структуры, — показать, что по 13–23 точкам годовой цикл FM не видит."""
    def f(c: Ctx):
        lh = _median_forecast(name, np.log(c.y[:, :c.T0 + 1]), max(c.H))
        return np.stack([np.exp(lh[:, h - 1]) for h in c.H], 1)
    return f
