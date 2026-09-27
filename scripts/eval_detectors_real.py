"""Детекторы на реальных данных: насколько часто каждый из них поднимает тревогу в МО-месяцах с задокументированным
событием (акт о ЧС по перечню МО или новость МЧС о реальном событии с названием МО) — при одинаковой для всех доле
тревог 3% в остальных МО-месяцах. Эталон построен по текстам, детекторы видят только траты; события, не задевшие
потребление, ограничивают полноту сверху для всех одинаково — важно сравнение, а не уровень.
Запуск: python scripts/eval_detectors_real.py [llm|regex]"""
import sys

import numpy as np
import pandas as pd

from ews import detect, panel as pnl, synth, text_features as tf
from ews.paths import OUTPUTS



def main(source="llm", fa=0.03):
    Y, ids, months, meta = pnl.load()
    Z = synth.h1_residuals(Y, months, synth.nat_series())
    D = detect.all_detectors(Z, synth.knn(meta))
    tx = tf.monthly(source=source)
    ev_local = np.nan_to_num(tx["act_local"]) > 0
    ev_news = np.nan_to_num(tx["news_mo"]) > 0
    lo, hi = months.index("2024-02"), months.index("2024-11")
    valid = ~np.isnan(Z[0])
    rows = []
    for name, S in D.items():
        row = {"детектор": name}
        for gname, G in [("акт по перечню МО", ev_local), ("новость МЧС с МО", ev_news), ("любой адресный текст", ev_local | ev_news)]:
            G = G.copy(); G[:, :lo] = False; G[:, hi + 1:] = False
            near = G | np.pad(G, ((0, 0), (1, 0)))[:, :-1]            # месяц события и следующий
            ctrl = valid & ~near
            thr = np.quantile(S[ctrl], 1 - fa)
            alarm = S > thr
            hit = [alarm[i, m] or (m + 1 < S.shape[1] and alarm[i, m + 1]) for i, m in zip(*np.where(G))]
            row[f"{gname}: событий"] = int(G.sum())
            row[f"{gname}: полнота@{int(fa * 100)}%"] = float(np.mean(hit)) if hit else np.nan
        rows.append(row)
    r = pd.DataFrame(rows).sort_values(f"любой адресный текст: полнота@{int(fa * 100)}%", ascending=False)
    base = 1 - (1 - fa) ** 2                                              # два месяца наугад при доле тревог fa
    pd.set_option("display.width", 250)
    print(f"источник новостей: {source}; случайный детектор поймал бы {base:.3f} (две попытки при {fa:.0%} тревог)")
    print(r.round(3).to_string(index=False))
    r.to_csv(OUTPUTS / "tables" / f"detectors_real_{source}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
