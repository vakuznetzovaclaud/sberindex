"""Онлайн-детекторы шока на нормированных остатках прогноза. Вход — z[категория, МО, месяц] (для месяцев без прогноза —
nan) и соседство МО; выход — оценка тревожности S[МО, месяц] (больше — тревожнее) для каждого детектора. Ищем падения,
поэтому работаем с -z. Все детекторы используют только прошлое и текущий месяц (тест tests/test_no_leakage.py).
"""
import numpy as np
from scipy.stats import rankdata


def _neg(z):
    return -np.nan_to_num(z, nan=0.0)


def threshold(z, cat=0):
    return _neg(z[cat])


def cusum(z, cat=0, k=0.5):
    x, s = _neg(z[cat]), np.zeros(z.shape[1])
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        s = np.maximum(0, s + x[:, t] - k); out[:, t] = s
    return out


def ewma(z, cat=0, lam=0.4):
    x, s = _neg(z[cat]), np.zeros(z.shape[1])
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        s = (1 - lam) * s + lam * x[:, t]; out[:, t] = s
    return out


def shiryaev_roberts(z, cat=0, delta=1.0):
    """Статистика Ширяева–Робертса для сдвига среднего -z на delta: R_t = (1 + R_{t-1}) * exp(delta * x_t - delta**2 / 2)."""
    x, r = _neg(z[cat]), np.zeros(z.shape[1])
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        r = (1 + r) * np.exp(delta * x[:, t] - delta ** 2 / 2); out[:, t] = np.log1p(r)
    return out


def glr(z, cat=None, window=3):
    """Обобщённое отношение правдоподобия с ограниченным окном (Lai 1995) для сдвига среднего вниз с неизвестной величиной:
    S_t = max по началу сдвига k от t - window + 1 до t величины max(0, sum(x[k..t]))**2 / (t - k + 1). Ловит и
    разовый провал, и ступеньку, не требуя задавать размер шока. cat=None — на сумме Стауффера по шести категориям."""
    x = _neg(z[cat]) if cat is not None else multicat_stouffer(z)
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        best = np.zeros(x.shape[0])
        for k in range(max(0, t - window + 1), t + 1):
            sm = np.maximum(0, x[:, k:t + 1].sum(1))
            best = np.maximum(best, sm ** 2 / (t - k + 1))
        out[:, t] = best
    return out


def multicat_max(z):
    return np.max(_neg(z), axis=0)


def multicat_stouffer(z):
    return _neg(z).sum(0) / np.sqrt(z.shape[0])


def cross_section(z, cat=0):
    """Ранг МО среди всех МО того же месяца (поперечный детектор)."""
    x = _neg(z[cat])
    return (x.argsort(0).argsort(0) + 1) / x.shape[0]


def spatial_scan(z, knn, ks=(1, 3, 5, 8), cat=None):
    """Скан по окрестностям: максимум по k нормированной суммы -z МО и k - 1 ближайших соседей (сумма / sqrt(k))."""
    x = _neg(z[cat]) if cat is not None else multicat_stouffer(z)
    return np.max([x[knn[:, :k]].sum(1) / np.sqrt(k) for k in ks], axis=0)


def bocpd(z, cat=0, hazard=1 / 12, delta=-1.0):
    """Байесовское онлайн-обнаружение (Adams, MacKay 2007) для нормированного ряда с двумя режимами среднего (0 и delta)
    — упрощение для коротких рядов: вероятность, что текущий режим — «шок»."""
    x = np.nan_to_num(z[cat], nan=0.0)
    p = np.zeros(x.shape[0])
    out = np.zeros_like(x)
    for t in range(x.shape[1]):
        prior = np.clip(p + (1 - p) * hazard, 1e-12, 1 - 1e-12)
        # в лог-шансах: при |z| в десятки σ обе плотности равны нулю в плавающей точке, отношение — нет
        logit = np.log(prior / (1 - prior)) + delta * x[:, t] - delta ** 2 / 2
        p = 1 / (1 + np.exp(-np.clip(logit, -700, 700)))
        out[:, t] = p
    return out


def with_text_prior(S, T, recall, false_rate, delta=1.5):
    """Слияние по Байесу: апостериорные лог-шансы шока = лог-отношение правдоподобия по тратам (сдвиг среднего на delta
    у оценки x в единицах её шума: delta * x - delta**2 / 2) + лог-отношение правдоподобия текстового сигнала T (0 или 1):
    тревога есть — log(recall / false_rate), тревоги нет — log((1 - recall) / (1 - false_rate)). Шум суммы Стауффера
    больше единицы (категории коррелируют, у z тяжёлые хвосты), поэтому S делится на робастное σ по всем МО того же
    месяца."""
    med = np.median(S, 0, keepdims=True)
    mad = 1.4826 * np.median(np.abs(S - med), 0, keepdims=True)
    x = (S - med) / np.where(mad > 0, mad, 1.0)
    llr_text = np.where(T > 0, np.log(recall / false_rate), np.log((1 - recall) / (1 - false_rate)))
    return delta * x - delta ** 2 / 2 + llr_text


def rank_ensemble(scores):
    """Ансамбль: среднее рангов оценок детекторов. Ранг — среди всех МО того же месяца, поэтому оценка месяца t не
    зависит от следующих месяцев."""
    return np.mean([rankdata(s, axis=0) / s.shape[0] for s in scores], axis=0)


def all_detectors(z, knn):
    """Все детекторы сравнения и два ансамбля: {название: S[МО, месяц]}."""
    d = {"порог z": threshold(z), "CUSUM": cusum(z), "EWMA": ewma(z), "Ширяев–Робертс": shiryaev_roberts(z),
         "BOCPD": bocpd(z), "поперечный ранг": cross_section(z), "6 категорий: максимум": multicat_max(z),
         "6 категорий: Стауффер": multicat_stouffer(z), "пространственный скан": spatial_scan(z, knn),
         "GLR (окно 3)": glr(z, cat=0), "6 категорий: GLR": glr(z)}
    d["ансамбль рангов"] = rank_ensemble([d[k] for k in ["порог z", "Ширяев–Робертс", "6 категорий: максимум", "пространственный скан"]])
    d["ансамбль: Стауффер + GLR + ранг"] = rank_ensemble([d[k] for k in ["6 категорий: Стауффер", "6 категорий: GLR", "поперечный ранг"]])
    return d
