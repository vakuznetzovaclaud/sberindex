"""Кластеры тревог на реальных данных 2024 г.: повторяется ли тревога в том же МО в следующем месяце и тянется ли она к
соседям — предпосылка моделей с самовозбуждением (процессы Хоукса). Полную модель Хоукса на 11 месяцах не оценить,
поэтому проверяются условные частоты. Тревога — сумма Стауффера по шести категориям (выбранный детектор) выше порога
3% по всем МО-месяцам 2024-02…2024-12. Соседи — 5 ближайших МО по центрам. Нулевая гипотеза — перестановки: для
соседей тревоги месяца перемешиваются между МО (число тревог месяца сохраняется), для повторения в том же МО —
месяцы внутри МО (частота тревог МО сохраняется).
Запуск: python scripts/eval_alarm_clusters.py"""
import numpy as np
import pandas as pd

from ews import detect, panel as pnl, synth
from ews.paths import OUTPUTS

FA, K, N_PERM = 0.03, 5, 500


def ratio(A, cond, prev=False):
    """P(тревога | условие) / P(тревога | нет условия); prev — условие берётся в прошлом месяце, а МО без собственной
    тревоги в прошлом месяце (так соседский вклад не смешивается с повторением в том же МО)."""
    a, c = (A[:, 1:], cond[:, :-1]) if prev else (A, cond)
    keep = ~A[:, :-1] if prev else np.ones_like(a, bool)
    return a[c & keep].mean() / a[~c & keep].mean()


def main():
    Y, ids, months, meta = pnl.load()
    Z = synth.h1_residuals(Y, months, synth.nat_series())
    lo, hi = months.index("2024-02"), months.index("2024-12")
    S = detect.multicat_stouffer(Z)[:, lo:hi + 1]
    ok = ~np.isnan(S).any(1)
    S, meta = S[ok], meta.loc[np.array(ids)[ok]]
    A = S > np.quantile(S, 1 - FA)
    nb = synth.knn(meta, K + 1)[:, 1:]
    reg = pd.factorize(meta["region_code"].astype(int))[0]
    R = np.eye(reg.max() + 1, dtype=int)[reg]                        # МО × регион
    rng = np.random.default_rng(42)

    def stats(A):
        nba = A[nb].any(1)                                            # хотя бы один сосед с тревогой [МО, месяц]
        same_reg = (R.T @ A.astype(int))[reg] - A > 0                  # другой МО того же региона с тревогой
        return {"повторение в том же МО через месяц": ratio(A[:, 1:], A[:, :-1]),
                "сосед с тревогой в том же месяце": ratio(A, nba),
                "сосед с тревогой в прошлом месяце": ratio(A, nba, prev=True),
                "другой МО региона с тревогой в том же месяце": ratio(A, same_reg)}

    obs = stats(A)
    null = {k: [] for k in obs}
    for _ in range(N_PERM):
        Pm = np.column_stack([rng.permutation(A[:, t]) for t in range(A.shape[1])])     # тревоги месяца — по другим МО
        Pt = np.array([rng.permutation(r) for r in A])                                 # месяцы внутри МО
        sp, st = stats(Pm), stats(Pt)
        for k in obs:
            null[k].append(st[k] if k.startswith("повторение") else sp[k])
    rows = [{"связь": k, "отношение частот": obs[k], "при перестановках": float(np.mean(null[k])),
             "p": float((1 + np.sum(np.array(null[k]) >= obs[k])) / (1 + N_PERM))} for k in obs]
    r = pd.DataFrame(rows)
    print(f"МО {len(A)}, месяцев {A.shape[1]}, тревог {int(A.sum())} ({A.mean():.1%})")
    print(r.round(3).to_string(index=False))
    r.to_csv(OUTPUTS / "tables" / "alarm_clusters.csv", index=False)


if __name__ == "__main__":
    main()
