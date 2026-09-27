"""Ретроспективная карта разладок: PELT (Killick et al. 2012, пакет ruptures) на всей истории отклонения МО от общего
фактора (логарифм, 24 месяца) по каждой категории; штраф BIC-типа 2·σ²·ln n, σ — шум МО по MAD первых разностей.
Сколько МО «ломается» в каждом месяце — всплески показывают сдвиги, общие для многих МО (в том числе в самих данных).
Запуск: python scripts/offline_breaks.py"""
import numpy as np
import pandas as pd
import ruptures as rpt

from ews import factor as fct, panel as pnl
from ews.paths import OUTPUTS


def breaks(x):
    d = np.diff(x)
    sigma = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    if not np.isfinite(sigma) or sigma <= 0:
        return []
    bk = rpt.Pelt(model="l2", min_size=3, jump=1).fit(x.reshape(-1, 1)).predict(pen=2 * sigma ** 2 * np.log(len(x)))
    return [b for b in bk if b < len(x)]


def main():
    Y, ids, months, meta = pnl.load()
    rows = []
    for c, cname in enumerate(pnl.CATS):
        LY = np.log(Y[c])
        U = LY - fct.factor(LY)
        for i in range(len(ids)):
            bk = breaks(U[i])
            edges = [0] + bk + [len(months)]
            for k, b in enumerate(bk):                  # скачок между соседними отрезками, а не между «всё до» и «всё после»
                jump = np.exp(U[i, b:edges[k + 2]].mean() - U[i, edges[k]:b].mean()) - 1
                rows.append({"категория": cname, "territory_id": ids[i], "месяц разладки": months[b], "сдвиг, %": 100 * jump})
    d = pd.DataFrame(rows)
    d.to_csv(OUTPUTS / "offline_breaks.csv", index=False)
    piv = d.pivot_table(index="месяц разладки", columns="категория", values="territory_id", aggfunc="count", fill_value=0)
    pd.set_option("display.width", 200)
    print("число МО с разладкой в месяце (PELT, отклонение от общего фактора):")
    print(piv.reindex(columns=pnl.CATS).to_string())


if __name__ == "__main__":
    main()
