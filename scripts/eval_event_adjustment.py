"""Поправка на событие (как поправки на акции и праздники в розничном прогнозе): для МО-месяцев с адресным текстовым
сигналом в первые 15 дней целевого месяца (акт о ЧС по перечню МО или новость МЧС о событии с названием МО) прогноз
умножается на exp(β), где β — сжатая медиана остатков таких же МО-месяцев прошлых точек (цели ≤ T0) за вычетом медианы
остатков остальных МО тех же месяцев. Всё, что использует β, известно в день прогноза. Запуск:
python scripts/eval_event_adjustment.py [модель, по умолчанию ens_main] [llm|regex]"""
import sys

import numpy as np
import pandas as pd

from ews import evaluate as ev, panel as pnl, text_features as tf
from ews.paths import OUTPUTS

K = 20          # сжатие: при n прошлых событиях доля доверия n / (n + K)


def main(model="ens_main", source="llm"):
    Y, ids, months, meta = pnl.load()
    txp = tf.monthly(source=source, day=15)
    flag = (np.nan_to_num(txp["act_local"]) > 0) | (np.nan_to_num(txp["news_mo"]) > 0)
    rows = []
    for cat, cname in enumerate(pnl.CATS):
        if model not in ev.available(cat):
            continue
        p = ev.load(model, cat)
        p = p[p.h == 1].reset_index(drop=True)
        a = Y[cat][p.i, p.target]
        r = np.log(a / p.yhat.to_numpy())
        f = flag[p.i, p.target]
        adj = p.yhat.to_numpy().copy()
        betas = []
        for T0 in sorted(p.origin.unique()):
            past = (p.target <= T0).to_numpy()
            n = int((past & f).sum())
            if n == 0:
                continue
            rel = np.median(r[past & f]) - np.median(r[past & ~f])
            beta = rel * n / (n + K)
            cur = (p.origin == T0).to_numpy() & f
            adj[cur] *= np.exp(beta)
            betas.append(beta)
        m = f & (p.origin > p.origin.min()).to_numpy()
        e0, e1 = np.abs(p.yhat.to_numpy() - a), np.abs(adj - a)
        rows.append({"категория": cname, "МО-месяцев с сигналом (h = 1)": int(m.sum()),
                     "MAE без поправки": e0[m].mean() if m.any() else np.nan, "MAE с поправкой": e1[m].mean() if m.any() else np.nan,
                     "изменение, %": 100 * (e1[m].mean() / e0[m].mean() - 1) if m.any() else np.nan,
                     "β последней точки, %": 100 * betas[-1] if betas else np.nan,
                     "MAE всех строк h = 1: изменение, %": 100 * (e1.mean() / e0.mean() - 1)})
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"модель: {model}; источник новостей: {source}")
    print(d.round(3).to_string(index=False))
    d.to_csv(OUTPUTS / "tables" / f"event_adjustment_{model}_{source}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
