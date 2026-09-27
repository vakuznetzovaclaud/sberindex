"""Чувствительность к сжатию сезонного отклонения МО (λ): панельная модель по всем категориям, LightGBM и ансамбль по
«Все категории». λ = 0,7 задано заранее, одно на все модели; здесь — что было бы при других значениях. Прогнозы
пересчитываются во временной папке и в outputs/forecasts не попадают.
Запуск: python scripts/eval_lambda.py"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ews import backtest, evaluate as ev, models, panel as pnl
from ews.paths import OUTPUTS

LAMS = (0.3, 0.5, 0.7, 1.0)
KEY = ["i", "origin", "h", "target"]


def main():
    Y, ids, months, meta = pnl.load()
    win = {"все точки": None, "разработка 01–06": [t for t in backtest.origins(months) if months[t] <= "2024-06"],
           "контроль 07–11": [t for t in backtest.origins(months) if months[t] >= "2024-07"]}

    def mae(p, c, o):
        """MAE прогнозов p категории c по точкам прогноза o (None — все точки)."""
        p = p if o is None else p[p.origin.isin(o)]
        return float(np.abs(p.yhat.to_numpy() - Y[c][p.i, p.target]).mean())

    fm = [ev.load(m, 0).set_index(KEY).yhat for m in ("timesfm25_sa", "tirex2_sa", "chronos2_sa")]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        backtest.DIR = Path(tmp)
        for lam in LAMS:
            panels = {c: backtest.run("panel", lambda x, lam=lam: models.panel(x, lam=lam), c) for c in range(len(pnl.CATS))}
            for c, p in panels.items():
                rows.append({"модель": "панель", "категория": pnl.CATS[c], "λ": lam, **{w: mae(p, c, o) for w, o in win.items()}})
            if lam in (0.5, 0.7):
                lg = backtest.run("lgbm", lambda x, lam=lam: models.lgbm_correction(x, lam=lam), 0)
                rows.append({"модель": "LightGBM", "категория": pnl.CATS[0], "λ": lam, **{w: mae(lg, 0, o) for w, o in win.items()}})
                # ансамбль: панель и LightGBM при этом λ, FM — как в основном прогоне (их вход сжат на 0,7)
                e = pd.concat([panels[0].set_index(KEY).yhat, lg.set_index(KEY).yhat] + fm, axis=1).dropna().mean(1).rename("yhat").reset_index()
                rows.append({"модель": "ансамбль (панель и LightGBM при этом λ)", "категория": pnl.CATS[0], "λ": lam,
                             **{w: mae(e, 0, o) for w, o in win.items()}})
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(r.round(1).to_string(index=False))
    r.to_csv(OUTPUTS / "tables" / "forecast_lambda.csv", index=False)


if __name__ == "__main__":
    main()
