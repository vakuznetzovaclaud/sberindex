"""Полусинтетическая проверка детекторов: в реальные ряды внедряются шоки известной формы, размера и времени, прогноз
панельной модели пересчитывается на искажённых данных, детекторы работают по остаткам. Формы шоков — из литературы о
карточных данных при стихийных бедствиях и из паводка 2024 г.: ступенька, провал с восстановлением, рампа, отраслевой
провал (маркетплейсы и транспорт)."""
import numpy as np

from . import factor as fct, national, panel as pnl
from .models import deviation_forecast

SHAPES = ("ступенька", "провал", "рампа", "отраслевой")
# индексы месяцев панели 2023-01…2024-12
FIRST_RESID = 13                # 2024-02: первый месяц с одношаговым остатком (прогноз из 2024-01)
SHOCK_FROM, SHOCK_TO = 15, 21   # 2024-04…2024-10: месяцы начала внедряемых шоков


def knn(meta, k=8):
    """k ближайших МО по центрам (первым — сам МО, даже если у соседа те же координаты)."""
    lat, lon = np.radians(meta["lat"].astype(float).to_numpy()), np.radians(meta["lon"].astype(float).to_numpy())
    xyz = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, -1.0)
    return np.argsort(d2, 1, kind="stable")[:, :k]


def zscore(R):
    """Остатки [МО × месяц] переводятся в z: минус медиана МО, делить на MAD МО (шум ряда); затем вычесть медиану z
    месяца по всем МО: общая для страны ошибка месяца (промах прогноза фактора) — не местный шок. Вычитать медиану
    именно из z, а не из сырых остатков: у крупных и малых МО разная доля общей ошибки. Нормировка по всему ряду
    видит будущее — годится для разметки задним числом (эталон системы тревог), не для детекторов."""
    med = np.nanmedian(R, 1, keepdims=True)
    mad = np.nanmedian(np.abs(R - med), 1, keepdims=True) * 1.4826
    Z = (R - med) / np.where(mad > 0, mad, np.nan)
    return Z - np.nanmedian(Z, 0, keepdims=True)


def zscore_online(R, U, first=FIRST_RESID, min_past=6):
    """Как zscore, но без будущего — как работал бы детектор в реальном времени: центр и шум МО в месяце t — по
    остаткам только прошлых месяцев (first…t-1). Центр — медиана прошлых остатков, если их не меньше трёх, иначе 0;
    шум — MAD прошлых остатков, если их не меньше min_past, иначе — по месячным изменениям отклонения МО за 2023 г.
    (известны в любой точке 2024 г.). Затем — минус медиана z того же месяца по всем МО (траты всех МО за месяц t
    становятся известны одновременно)."""
    d = np.diff(U[:, :12], axis=1)
    s0 = 1.4826 * np.nanmedian(np.abs(d - np.nanmedian(d, 1, keepdims=True)), 1)
    Z = np.full(R.shape, np.nan)
    for t in range(first, R.shape[1]):
        past = R[:, first:t]
        n = np.sum(~np.isnan(past), 1)
        med = np.where(n >= 3, np.nanmedian(past, 1) if past.shape[1] else 0, 0)
        mad = np.where(n >= min_past, 1.4826 * np.nanmedian(np.abs(past - med[:, None]), 1) if past.shape[1] else s0, s0)
        Z[:, t] = (R[:, t] - med) / np.where(mad > 0, mad, np.nan)
    return Z - np.nanmedian(Z, 0, keepdims=True)


def h1_residuals(Y, months, nat_by_cat, lam=0.7, online=True):
    """Нормированные остатки одношагового прогноза панельной модели для всех категорий: z[кат, МО, месяц] (nan до 2024-02).
    online — нормировка только по прошлому (для детекторов); иначе — по всему ряду (для разметки задним числом)."""
    C, N, T = Y.shape
    Z = np.full(Y.shape, np.nan)
    for c in range(C):
        LY = np.log(Y[c]); F = fct.factor(LY); U = LY - F
        R = np.full((N, T), np.nan)
        for t in range(FIRST_RESID, T):
            fh = fct.forecast(F, t - 1, [1], nat_by_cat[c], months)[0]
            R[:, t] = LY[:, t] - fh - deviation_forecast(U, t - 1, [1], lam)[:, 0]
        Z[c] = zscore_online(R, U) if online else zscore(R)
    return Z


def inject(Y, rng, shape, size, knn_idx, n_events=60, clustered=False, months_from=SHOCK_FROM, months_to=SHOCK_TO):
    """Возвращает искажённую панель и метки: tau[МО] — месяц начала шока (-1 — без шока)."""
    Y = Y.copy()
    C, N, T = Y.shape
    tau = np.full(N, -1)
    for s in rng.choice(N, n_events, replace=False):
        t0 = int(rng.integers(months_from, months_to + 1))
        for i in (knn_idx[s, :5] if clustered else [s]):
            if tau[i] >= 0:
                continue
            tau[i] = t0
            prof = np.zeros(T)
            if shape == "ступенька":
                prof[t0:] = size
            elif shape == "рампа":
                prof[t0:] = np.minimum(size, size * (np.arange(T - t0) + 1) / 3)
            else:
                prof[t0] = size; prof[t0 + 1:t0 + 2] = size / 2
            if shape == "отраслевой":                 # падают маркетплейсы и транспорт, «Все категории» — на их долю
                share = (Y[4, i] + Y[5, i]) / Y[0, i]
                Y[4, i] *= (1 - prof); Y[5, i] *= (1 - prof); Y[0, i] *= (1 - prof * share)
            else:
                Y[:, i] *= (1 - prof)
    return Y, tau


def nat_series():
    return [national.for_category(c) for c in pnl.CATS]
