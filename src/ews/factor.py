"""Общий фактор месяца и его прогноз.
Фактор f[t] — медиана логарифма трат на жителя по МО в месяце t: общая для всех МО сезонность и тренд.
Прогноз фактора — главный рычаг точности (как «общий множитель» у победителей Kaggle GoDaddy и M5): прирост г/г
оценивается равной смесью прироста по самой панели (1–3 последних месяца) и по национальному ряду СберИндекса за
3 месяца (ряд с 2018 г. — оценка устойчива и в начале 2024 г., когда по панели есть лишь 1–3 месяца прироста)."""
import numpy as np
import pandas as pd

RECENT = 3          # месяцев для уровня и прироста г/г; в начале 2024 г. меньше: панель начинается с января 2023 г.


def recent_k(T0):
    """Сколько последних месяцев по T0 включительно брать для уровня и прироста г/г: RECENT, но не больше, чем
    месяцев, у которых в панели есть тот же месяц годом раньше."""
    return min(RECENT, T0 - 11)


def recent_and_last_year(U, T0):
    """Средний уровень строк U за k = recent_k(T0) последних месяцев по T0 включительно и за те же месяцы год назад."""
    k = recent_k(T0)
    return U[:, T0 - k + 1:T0 + 1].mean(1), U[:, T0 - k - 11:T0 - 11].mean(1)


def factor(LY):
    return np.median(LY, axis=0)


def growth_panel(F, T0):
    """Прирост фактора г/г: среднее помесячных приростов за последние k = recent_k(T0) месяцев."""
    return float(np.mean([F[T0 - j] - F[T0 - j - 12] for j in range(recent_k(T0))]))


def growth_national(nat, months, T0, k=RECENT):
    """Средний лог-прирост г/г национального ряда за k месяцев по T0 включительно (известен на момент T0)."""
    p = pd.Period(months[T0], "M")
    return float(np.mean([np.log(nat[str(p - j)]) - np.log(nat[str(p - j - 12)]) for j in range(k)]))


def forecast(F, T0, horizons, nat=None, months=None, w_nat=0.5):
    g = growth_panel(F, T0)
    if nat is not None:
        g = (1 - w_nat) * g + w_nat * growth_national(nat, months, T0)
    return np.array([F[T0 + h - 12] + g for h in horizons])
