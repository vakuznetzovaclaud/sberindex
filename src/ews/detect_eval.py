"""Метрики детекторов на полусинтетике: полнота при фиксированной доле ложных тревог, средняя задержка, площадь под
кривой точность–полнота с допуском по времени (аналог VUS-PR, Liu & Paparrizos 2024), лучший по порогу F1 с тем же
допуском, NAB-подобная оценка с наградой за раннее срабатывание (Lavin & Ahmad 2015)."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


FA_CURVE = (0.005, 0.01, 0.02, 0.03, 0.05, 0.1)            # сетка долей ложных тревог для кривой «задержка — ложные тревоги»


def evaluate(S, tau, months_eval, max_delay=2, fa_levels=FA_CURVE):
    """S — оценки [МО, месяц] (для месяцев months_eval), tau — месяц шока или -1. На каждой доле ложных тревог:
    полнота, средняя задержка пойманных и задержка с учётом пропусков (пропуск = max_delay + 1 месяц) — одна
    монотонная величина для кривой «задержка — ложные тревоги», как в InDiD (Romanenkova et al., 2022)."""
    T = np.array(months_eval)
    s = S[:, T]
    shocked = tau >= 0
    clean = np.ones_like(s, bool)
    for i in np.where(shocked)[0]:
        clean[i, T >= tau[i]] = False
    res = {}
    for fa in fa_levels:
        thr = np.quantile(s[clean], 1 - fa)
        det, delays = 0, []
        for i in np.where(shocked)[0]:
            w = (T >= tau[i]) & (T <= tau[i] + max_delay)
            hit = np.where(s[i, w] > thr)[0]
            if len(hit):
                det += 1; delays.append(hit[0])
        k = f"{fa * 100:g}%"
        res[f"полнота@{k}"] = det / max(shocked.sum(), 1)
        res[f"задержка@{k}"] = float(np.mean(delays)) if delays else np.nan
        res[f"задержка с пропусками@{k}"] = (sum(delays) + (max_delay + 1) * (shocked.sum() - det)) / max(shocked.sum(), 1)
    aps = []
    for buf in range(max_delay + 1):                  # «объём»: средняя AP по допускам 0…max_delay месяцев
        y = np.zeros_like(s, bool)
        for i in np.where(shocked)[0]:
            y[i, (T >= tau[i]) & (T <= tau[i] + buf)] = True
        mask = clean | y
        aps.append(average_precision_score(y[mask], s[mask]))
    res["VUS-PR"] = float(np.mean(aps))
    prec, rec, _ = precision_recall_curve(y[mask], s[mask])        # y — с допуском max_delay месяцев после начала шока
    res["F1 с допуском"] = float(np.max(2 * prec * rec / np.maximum(prec + rec, 1e-12)))
    res["NAB-подобная"] = nab_score(s, T, tau, shocked, clean, max_delay)
    return res


def nab_score(s, T, tau, shocked, clean, max_delay=2, a_fp=0.11, quantiles=np.linspace(0.90, 0.999, 60)):
    """NAB-подобная оценка по профилю «стандартный» (Lavin & Ahmad 2015): срабатывание в окне шока даёт награду, тем
    большую, чем раньше (1 в месяц шока, 0,5 через месяц, 0,25 через два), пропуск — -1, ложная тревога — -0,11.
    Как в NAB, порог подбирается для каждого детектора, оценка нормируется: 0 — детектор, который молчит, 100 — идеальный."""
    n = int(shocked.sum())
    if n == 0:
        return np.nan
    best = -float(n)                                  # молчащий детектор: все шоки пропущены
    for q in quantiles:
        thr = np.quantile(s[clean], q)
        reward = 0.0
        for i in np.where(shocked)[0]:
            w = np.where((T >= tau[i]) & (T <= tau[i] + max_delay))[0]
            hit = [k for k, j in enumerate(w) if s[i, j] > thr]
            reward += 0.5 ** hit[0] if hit else -1.0
        best = max(best, reward - a_fp * (s[clean] > thr).sum())
    return 100 * (best + n) / (2 * n)
