"""Скользящий бэктест: последний наблюдённый месяц 2024-01…2024-11, горизонты 1–3, все МО и шесть категорий.
Прогнозы сохраняются по модели и категории; метрики считаются на общих строках."""
import numpy as np
import pandas as pd

from . import national, panel as pnl, text_features
from .models import Ctx
from .paths import OUTPUTS, config

DIR = OUTPUTS / "forecasts"


def origins(months):
    p = config()["period"]
    return [i for i, m in enumerate(months) if p["origins_from"] <= m <= p["origins_to"]]


def run(name, fn, cat=0, subset=None, text=None, nowcast_day=None):
    """text — источник новостей для текстовых признаков («llm» или «regex»); None — без текста.
    nowcast_day — режим наукаста: добавить тексты первых nowcast_day дней месяца после точки прогноза."""
    Y, ids, months, meta = pnl.load()
    tx = txp = None
    if text:
        tx = text_features.monthly(source=text)
        if nowcast_day:
            txp = text_features.monthly(source=text, day=nowcast_day)
    nat = national.for_category(pnl.CATS[cat])
    H = config()["period"]["horizons"]
    rows = []
    sub = np.arange(len(ids)) if subset is None else np.asarray(subset)
    for T0 in origins(months):
        P = np.full((len(ids), len(H)), np.nan)
        P[sub] = fn(Ctx(y=Y[cat][sub], T0=T0, H=H, months=months, nat=nat, meta=meta.iloc[sub], ya=Y[:, sub], cat=cat,
                       tx={k: v[sub] for k, v in tx.items()} if tx else None,
                       txp={k: v[sub] for k, v in txp.items()} if txp else None))
        for j, h in enumerate(H):
            if T0 + h < len(months):
                rows.append(pd.DataFrame({"i": np.arange(len(ids)), "origin": T0, "h": h, "target": T0 + h, "yhat": P[:, j]}))
    out = pd.concat(rows).dropna(subset=["yhat"])
    DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DIR / f"{name}__c{cat}.parquet")
    return out


def stratified_subset(n=400, seed=42):
    """Стратифицированная по численности выборка МО (квинтили) — для медленных моделей по отдельным рядам."""
    _, ids, _, meta = pnl.load()
    q = pd.qcut(meta["population"].rank(method="first"), 5, labels=False).to_numpy()
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.where(q == k)[0], n // 5, replace=False) for k in range(5)]))
