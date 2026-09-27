"""Сравнение детекторов на полусинтетике: формы × размеры × одиночные/кластерные шоки, несколько повторов.
Запуск: python scripts/eval_detectors.py [повторов]"""
import sys

import numpy as np
import pandas as pd

from ews import detect, detect_eval, panel as pnl, synth
from ews.paths import OUTPUTS


# Текстовый сигнал в полусинтетике: тревога в месяц шока с вероятностью recall и ложные тревоги с частотой false_rate
# в остальных МО-месяцах. Сценарии задают качество текста, которое понадобилось бы детектору; измеренное качество
# новостей МЧС как тревоги по отдельному МО ниже самого слабого сценария (раздел 10 отчёта: подъём около 1).
TEXT_SCENARIOS = {"текст слабый": (0.2, 0.09), "текст хороший": (0.5, 0.05), "текст идеальный": (0.8, 0.02)}


def text_signal(tau, shape, rng, recall, false_rate):
    T = (rng.random(shape) < false_rate).astype(float)
    for i in np.where(tau >= 0)[0]:
        T[i, tau[i]] = float(rng.random() < recall)
    return T


def main(reps=3):
    Y0, ids, months, meta = pnl.load()
    nat = synth.nat_series()
    kn = synth.knn(meta)
    T_eval = list(range(synth.FIRST_RESID, len(months)))
    rows = []
    rng = np.random.default_rng(42)
    for shape in synth.SHAPES:
        for size in (0.05, 0.10, 0.20):
            for clustered in (False, True):
                for rep in range(reps):
                    Y, tau = synth.inject(Y0, rng, shape, size, kn, clustered=clustered)
                    Z = synth.h1_residuals(Y, months, nat)
                    D = detect.all_detectors(Z, kn)
                    for sc, (rec, fr) in TEXT_SCENARIOS.items():
                        T = text_signal(tau, D["порог z"].shape, rng, rec, fr)
                        D[f"Стауффер + {sc}"] = detect.with_text_prior(D["6 категорий: Стауффер"], T, rec, fr)
                    for name, S in D.items():
                        rows.append({"форма": shape, "размер": size, "кластер": clustered, "повтор": rep, "детектор": name}
                                    | detect_eval.evaluate(S, tau, T_eval))
                print(shape, size, clustered, flush=True)
    res = pd.DataFrame(rows)
    OUTPUTS.mkdir(exist_ok=True)
    res.to_csv(OUTPUTS / "detectors_synthetic.csv", index=False)
    summ = res.groupby("детектор")[["VUS-PR", "F1 с допуском", "полнота@3%", "задержка@3%", "NAB-подобная"]].mean().sort_values("VUS-PR", ascending=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
