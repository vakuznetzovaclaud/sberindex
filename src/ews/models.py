"""Модели прогноза трат на жителя МО. Каждая модель получает контекст точки прогноза и возвращает матрицу
[МО × горизонт]. Общий принцип сильных моделей: log y = общий фактор месяца + отклонение МО (сезонность МО + уровень)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import factor as fct
from .factor import recent_and_last_year


@dataclass
class Ctx:
    y: np.ndarray            # МО × месяц, одна категория
    T0: int                  # индекс последнего наблюдённого месяца
    H: list                  # горизонты
    months: list             # подписи месяцев 'ГГГГ-ММ'
    nat: object = None       # национальный ряд (pandas.Series по 'ГГГГ-ММ') или None
    meta: object = None      # атрибуты МО
    ya: np.ndarray = None    # все категории [кат × МО × месяц] — для глобальных моделей; y == ya[cat]
    cat: int = 0
    tx: dict = None          # текстовые признаки {имя: [МО × месяц]} с датой знания (text_features.monthly)
    txp: dict = None         # наукаст: признаки месяца T0+1 по текстам его первых дней (известны в день прогноза)

    @property
    def LY(self):
        return np.log(self.y)


LGBM = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, objective="l1", verbose=-1, random_state=42)


def sa_deviation(U, T0, lam=0.7):
    """Отклонение МО до T0 без его сезонного профиля за 2023 г. (профиль сжат к нулю на lam) и сам профиль
    [МО × месяц года]. Профиль 2023 г. известен в любой точке прогноза 2024 г."""
    s = lam * (U[:, :12] - U[:, :12].mean(1, keepdims=True))
    return U[:, :T0 + 1] - s[:, np.arange(T0 + 1) % 12], s


def _decompose(c):
    LY = c.LY
    F = fct.factor(LY)
    return F, LY - F


def snaive(c):
    return np.stack([c.y[:, c.T0 + h - 12] for h in c.H], 1)


def snaive_growth(c):
    k = fct.recent_k(c.T0)
    g = c.y[:, c.T0 - k + 1:c.T0 + 1].sum(1) / c.y[:, c.T0 - k - 11:c.T0 - 11].sum(1)
    return np.stack([c.y[:, c.T0 + h - 12] * g for h in c.H], 1)


def deviation_forecast(U, T0, H, lam=0.7):
    """Отклонение МО от фактора: уровень последних месяцев + сезонное отклонение МО прошлого года, сжатое к общему (lam)."""
    u_rec, u_ly = recent_and_last_year(U, T0)
    return np.stack([u_rec + lam * (U[:, T0 + h - 12] - u_ly) for h in H], 1)


def factor_forecast(c, use_national=True):
    F, _ = _decompose(c)
    return fct.forecast(F, c.T0, c.H, c.nat if use_national else None, c.months)


def panel(c, lam=0.7, use_national=True):
    F, U = _decompose(c)
    return np.exp(factor_forecast(c, use_national)[None, :] + deviation_forecast(U, c.T0, c.H, lam))


def _text(c, t0):
    """Текстовые признаки месяца t0 (известны к концу t0) и, в режиме наукаста, первых дней месяца t0 + 1."""
    cols = {k: v[:, t0] for k, v in c.tx.items()} if c.tx else {}
    if c.txp:
        cols |= {f"nowcast_{k}": v[:, t0 + 1] if t0 + 1 < v.shape[1] else np.zeros(len(v)) for k, v in c.txp.items()}
    return cols


def _log_population(meta):
    """Логарифм населения МО; пропуски — медианой."""
    return np.log(meta["population"].fillna(meta["population"].median()).to_numpy())


def _deviation_features(U, t0, h, months, pop, reg):
    """Признаки поправки к отклонению МО, общие для обоих LightGBM: прирост г/г, сезонное отклонение месяца цели
    год назад, последние изменения отклонения, горизонт, месяц цели, население, регион. Возвращает признаки, средний
    уровень последних месяцев и тех же месяцев годом раньше."""
    u_rec, u_ly = recent_and_last_year(U, t0)
    X = pd.DataFrame({"yoy": u_rec - u_ly, "season_ly": U[:, t0 + h - 12] - u_ly, "d1": U[:, t0] - U[:, t0 - 1],
                      "d12": U[:, t0] - U[:, t0 - 12], "season_shift": U[:, t0 + h - 12] - U[:, t0 - 12], "h": float(h),
                      "month": float((int(months[t0][-2:]) + h - 1) % 12 + 1), "log_pop": pop, "region": reg})
    return X, u_rec, u_ly


def lgbm_correction(c, lam=0.7, use_national=True):
    """Глобальный LightGBM учит поправку к отклонению панельной модели по всем МО и прошлым точкам прогноза.
    Если в контексте есть текстовые признаки (c.tx), они добавляются к признакам точки прогноза. Модель обучается,
    когда накоплено хотя бы два прошлых блока «точка × горизонт» (с точки 2024-03); в 2024-01 и 2024-02 прогноз равен
    панельному."""
    import lightgbm as lgb
    F, U = _decompose(c)
    pop, reg = _log_population(c.meta), c.meta["region_code"].astype(int).to_numpy()

    def feats(t0, h):
        X, u_rec, u_ly = _deviation_features(U, t0, h, c.months, pop, reg)
        return X.assign(**_text(c, t0)), u_rec + lam * (U[:, t0 + h - 12] - u_ly)

    Xs, ys = [], []
    for t0 in range(12, c.T0):
        for h in c.H:
            if t0 + h <= c.T0:
                X, base = feats(t0, h)
                Xs.append(X); ys.append(U[:, t0 + h] - base)
    model = None
    if len(Xs) >= 2:
        model = lgb.LGBMRegressor(**LGBM).fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                                              categorical_feature=["region"])
    fh = factor_forecast(c, use_national)
    out = []
    for j, h in enumerate(c.H):
        X, base = feats(c.T0, h)
        out.append(np.exp(fh[j] + base + (model.predict(X) if model is not None else 0.0)))
    return np.stack(out, 1)


def lgbm_global(c, lam=0.7, use_national=True):
    """Глобальный LightGBM сразу по шести категориям (как глобальные модели победителей M5): учит поправку к отклонению
    панельной модели. Кроме собственной истории отклонения МО — номер категории, прирост г/г категории «Все категории»
    того же МО и медианы по региону (общий для региона сдвиг заметнее, чем в шумном ряду отдельного МО).
    Модель обучается один раз на точку прогноза и даёт поправку для всех горизонтов."""
    import lightgbm as lgb
    ya, kt = c.ya, c.cat
    Us = [_decompose(Ctx(y=ya[k], T0=c.T0, H=c.H, months=c.months))[1] for k in range(len(ya))]
    pop, reg = _log_population(c.meta), c.meta["region_code"].astype(int).to_numpy()
    _, inv = np.unique(reg, return_inverse=True)
    groups = [np.where(inv == r)[0] for r in range(inv.max() + 1)]

    def reg_median(v):
        out = np.empty_like(v)
        for g in groups:
            out[g] = np.median(v[g])
        return out

    def feats(kc, t0, h):
        U = Us[kc]
        X, _, u_ly = _deviation_features(U, t0, h, c.months, pop, reg)
        g = X["yoy"].to_numpy()
        X = X.assign(category=float(kc), reg_yoy=reg_median(g), reg_d12=reg_median(X["d12"].to_numpy()),
                     yoy_all=np.subtract(*recent_and_last_year(Us[0], t0)), **_text(c, t0))
        return X, u_ly + g + lam * (U[:, t0 + h - 12] - u_ly)        # прошлогодний уровень + прирост г/г + сезонность

    Xs, ys = [], []
    for kc in range(len(ya)):
        for t0 in range(12, c.T0):
            for h in c.H:
                if t0 + h <= c.T0:
                    X, base = feats(kc, t0, h)
                    Xs.append(X); ys.append(Us[kc][:, t0 + h] - base)
    model = None
    if len(Xs) >= 2:
        model = lgb.LGBMRegressor(**LGBM).fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys),
                                              categorical_feature=["region", "category"])
    fh = factor_forecast(c, use_national)
    out = []
    for j, h in enumerate(c.H):
        X, base = feats(kt, c.T0, h)
        out.append(np.exp(fh[j] + base + (model.predict(X) if model is not None else 0.0)))
    return np.stack(out, 1)


def _ses(x, alphas=np.linspace(0.05, 0.95, 19)):
    """Простое экспоненциальное сглаживание, векторно по рядам: α каждого ряда — по минимуму одношаговых ошибок в
    выборке. Возвращает последний уровень."""
    best_sse, best_lvl = np.full(len(x), np.inf), x[:, -1].copy()
    for a in alphas:
        lvl, sse = x[:, 0].copy(), np.zeros(len(x))
        for t in range(1, x.shape[1]):
            sse += (x[:, t] - lvl) ** 2
            lvl = a * x[:, t] + (1 - a) * lvl
        better = sse < best_sse
        best_sse[better], best_lvl[better] = sse[better], lvl[better]
    return best_lvl


def stat_on_seasonally_adjusted(method="theta", lam=0.7):
    """Классические методы на том же входе, что и FM (отклонение МО без его профиля 2023 г.): «наив» — последнее
    значение, «ses» — экспоненциальное сглаживание, «theta» — сглаживание плюс половина линейного тренда (метод Theta,
    Assimakopoulos & Nikolopoulos 2000; победитель M3). Нужны, чтобы отделить вклад FM от вклада сезонной коррекции."""
    def f(c):
        F, U = _decompose(c)
        usa, s = sa_deviation(U, c.T0, lam)
        if method == "naive":
            lvl, slope = usa[:, -1], np.zeros(len(usa))
        else:
            lvl = _ses(usa)
            t = np.arange(usa.shape[1])
            slope = ((t - t.mean()) * (usa - usa.mean(1, keepdims=True))).sum(1) / ((t - t.mean()) ** 2).sum()
            slope = slope / 2 if method == "theta" else np.zeros(len(usa))
        fh = factor_forecast(c)
        return np.stack([np.exp(fh[j] + lvl + slope * h + s[:, (c.T0 + h) % 12]) for j, h in enumerate(c.H)], 1)
    return f
