"""Что даёт текст прогнозу: MAE моделей с текстовыми признаками и без них на всех строках и на МО-месяцах, где текст
что-то сообщает. Запуск: python scripts/eval_text_value.py
Строки «сигнал в точке прогноза» — в месяце T0 для МО есть акт о ЧС или новость МЧС об ущербе с упоминанием МО;
«событие в целевом месяце» — такой текст появился в месяце, который прогнозируем (прогноз его ещё не мог знать)."""
import numpy as np
import pandas as pd

from ews import evaluate as ev, panel as pnl, text_features
from ews.paths import OUTPUTS

PAIRS = [("lgbm", "lgbm_text"), ("lgbm", "lgbm_nowcast_text"), ("lgbm", "lgbm_text_regex")]


def main():
    Y, ids, months, meta = pnl.load()
    tx = text_features.monthly()
    sig = (np.nan_to_num(tx["act_recent"]) > 0) | (np.nan_to_num(tx["news_mo"]) > 0)
    txp = text_features.monthly(day=15)
    sigp = (np.nan_to_num(txp["act_new"]) > 0) | (np.nan_to_num(txp["news_mo"]) > 0)
    rows = []
    for cat, cname in enumerate(pnl.CATS):
        for base, text in PAIRS:
            if not all(n in ev.available(cat) for n in (base, text)):
                continue
            a, b = ev.load(base, cat), ev.load(text, cat)
            y = Y[cat][a.i, a.target]
            ea, eb = np.abs(a.yhat.to_numpy() - y), np.abs(b.yhat.to_numpy() - y)
            h1 = (a.h == 1).to_numpy()
            masks = {"все строки": np.ones(len(a), bool), "сигнал в точке прогноза": sig[a.i, a.origin],
                     "событие в целевом месяце": sig[a.i, a.target],
                     "h = 1, текст первых 15 дней целевого месяца": h1 & sigp[a.i, np.minimum(a.origin + 1, sigp.shape[1] - 1)]}
            for mname, m in masks.items():
                g = pd.DataFrame({"t": a.target[m], "a": ea[m], "b": eb[m]}).groupby("t").mean()
                d = g.b - g.a
                rows.append({"категория": cname, "модель": base, "с текстом": text, "строки": mname, "n": int(m.sum()),
                             "MAE без текста": ea[m].mean(), "MAE с текстом": eb[m].mean(),
                             "изменение, %": 100 * (eb[m].mean() / ea[m].mean() - 1),
                             "доля строк лучше": float((eb[m] < ea[m]).mean()),
                             "DM p (по месяцам)": ev.dm_test(g.b.values, g.a.values, 1) if len(g) >= 3 else np.nan,
                             "месяцев с выигрышем": f"{int((d < 0).sum())}/{len(d)}"})
    res = pd.DataFrame(rows)
    res.to_csv(OUTPUTS / "tables" / "text_value.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
