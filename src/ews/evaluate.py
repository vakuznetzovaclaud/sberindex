"""Метрики прогноза: MAE (главная), MAPE, WAPE, MASE, R² уровня, R² прироста г/г, R²_oos к сезонному наиву, доля МО,
где модель лучше базовой, и тест Диболда–Мариано с поправкой Harvey–Leybourne–Newbold по месяцам-целям."""
import numpy as np
import pandas as pd
from scipy import stats

from . import panel as pnl, synth
from .backtest import DIR


def load(name, cat=0):
    return pd.read_parquet(DIR / f"{name}__c{cat}.parquet").sort_values(["origin", "h", "i"]).reset_index(drop=True)


def available(cat=0):
    return sorted(f.name.split("__c")[0] for f in DIR.glob(f"*__c{cat}.parquet"))


def h1_log_residuals(name, cat=0):
    """Лог-остаток одношагового прогноза модели: r[МО, месяц] = log факт - log прогноз (h = 1); nan вне месяцев
    с прогнозом."""
    Y = pnl.load()[0][cat]
    p = load(name, cat)
    p = p[p.h == 1]
    r = np.full(Y.shape, np.nan)
    r[p.i.to_numpy(), p.target.to_numpy()] = np.log(Y[p.i, p.target]) - np.log(p.yhat.to_numpy())
    return r


def h1_response(model, cat=0, scale=False):
    """Отклик МО R[МО, месяц] в %: лог-остаток одношагового прогноза минус его медиана по всем МО того же месяца (общая
    ошибка месяца — не местный отклик); nan вне месяцев с прогнозом. scale=True — в единицах собственного шума МО
    (synth.zscore): малые шумные МО не заглушают сигнал крупных."""
    r = h1_log_residuals(model, cat)
    return synth.zscore(r) if scale else 100 * (r - np.nanmedian(r, 0, keepdims=True))


def ensemble(members, cat=0, name=None, how="mean"):
    """Ансамбль с равными весами (среднее или медиана прогнозов участников) на общих строках; сохраняется как модель."""
    P = [load(m, cat).set_index(["i", "origin", "h", "target"]).yhat.rename(m) for m in members]
    df = pd.concat(P, axis=1, join="inner")
    out = (df.mean(1) if how == "mean" else df.median(1)).rename("yhat").reset_index()
    name = name or "ens_" + "+".join(members)
    out.to_parquet(DIR / f"{name}__c{cat}.parquet")
    return name


def ensemble_online(members, cat=0, name=None):
    """Ансамбль с весами, обратными MAE участника по уже наблюдённым к точке прогноза целям (target ≤ T0) прошлых точек;
    в первой точке веса равные. Веса не используют ничего после T0."""
    Y, _, _, _ = pnl.load()
    P = {m: load(m, cat).set_index(["i", "origin", "h", "target"]).yhat.rename(m) for m in members}
    df = pd.concat(P.values(), axis=1, join="inner").reset_index()
    y = Y[cat][df.i, df.target]
    ae = {m: np.abs(df[m].to_numpy() - y) for m in members}
    out = []
    for T0 in sorted(df.origin.unique()):
        seen = (df.target <= T0).to_numpy()
        w = np.array([1 / ae[m][seen].mean() if seen.any() else 1.0 for m in members])
        w = w / w.sum()
        cur = df.origin.to_numpy() == T0
        out.append(df.loc[cur, ["i", "origin", "h", "target"]].assign(yhat=df.loc[cur, members].to_numpy() @ w))
    name = name or "ens_online_" + "+".join(members)
    pd.concat(out).to_parquet(DIR / f"{name}__c{cat}.parquet")
    return name


def dm_test(e1, e2, h):
    """Диболд–Мариано для рядов потерь (по месяцам-целям), поправка HLN на горизонт h. Если оценка дисперсии с
    автоковариациями до лага h - 1 отрицательна (бывает на коротких рядах), тест повторяется с h = 1 — как в
    forecast::dm.test (R)."""
    d = np.asarray(e1) - np.asarray(e2)
    n = len(d)
    if n < 3:
        return np.nan
    gamma = [np.sum((d[k:] - d.mean()) * (d[:n - k] - d.mean())) / n for k in range(h)]   # автоковариации, как acf в R
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return dm_test(e1, e2, 1) if h > 1 else np.nan
    stat = d.mean() / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return 2 * stats.t.sf(abs(stat), n - 1)


def table(names, cat=0, base="snaive", ref=None, origins=None):
    """Метрики на общих строках всех моделей; origins — список индексов точек прогноза (окно оценки) или None."""
    Y, ids, months, meta = pnl.load()
    y = Y[cat]
    P = {n: load(n, cat) for n in names}
    if origins is not None:
        P = {n: p[p.origin.isin(origins)].reset_index(drop=True) for n, p in P.items()}
    common = set.intersection(*[set(map(tuple, p[["i", "origin", "h"]].to_numpy())) for p in P.values()])
    for n in P:
        k = P[n][["i", "origin", "h"]].apply(tuple, axis=1).isin(common)
        P[n] = P[n][k].reset_index(drop=True)
    any_p = next(iter(P.values()))
    a = y[any_p.i, any_p.target]
    ly = y[any_p.i, any_p.target - 12]
    scale = np.array([np.mean(np.abs(np.diff(y[i, :o + 1]))) for i, o in zip(any_p.i, any_p.origin)])
    rows = []
    for n, p in P.items():
        f = p.yhat.to_numpy()
        e = np.abs(a - f)
        g, gh = a / ly - 1, f / ly - 1
        row = {"модель": n, "MAE": e.mean(), "MAPE, %": 100 * (e / a).mean(), "WAPE, %": 100 * e.sum() / a.sum(),
               "MASE": (e / scale).mean(), "R² уровня": 1 - ((a - f) ** 2).sum() / ((a - a.mean()) ** 2).sum(),
               "R² г/г": 1 - ((g - gh) ** 2).sum() / ((g - g.mean()) ** 2).sum()}
        if base in P:
            fb = P[base].yhat.to_numpy()
            row["R²_oos к наиву"] = 1 - ((a - f) ** 2).sum() / ((a - fb) ** 2).sum()
        if ref in P and n != ref:
            er = np.abs(a - P[ref].yhat.to_numpy())
            per = pd.DataFrame({"i": p.i, "d": e - er}).groupby("i").d.mean()
            row[f"доля МО лучше {ref}"] = (per < 0).mean()
            pv = []
            for h in sorted(p.h.unique()):
                m = p.h.to_numpy() == h
                l1 = pd.Series(e[m]).groupby(p.target.to_numpy()[m]).mean()
                l2 = pd.Series(er[m]).groupby(p.target.to_numpy()[m]).mean()
                pv.append(dm_test(l1.values, l2.values, h))
            row["DM p (h=1/2/3)"] = "/".join(f"{x:.3f}" for x in pv)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE")


def conformal(name, cat=0, levels=(0.8, 0.95), strata=5):
    """Интервалы прогноза без предположений о распределении (скользящий сплит-конформный метод): в точке T0 квантили
    лог-остатков той же модели и горизонта по целям, уже наблюдённым к T0 (target ≤ T0), отдельно по квинтилям
    численности МО. Возвращает покрытие, относительную ширину и интервальную оценку WIS (Bracher et al. 2021)
    по горизонтам; точки без истории остатков пропускаются."""
    Y, _, _, meta = pnl.load()
    p = load(name, cat)
    a = Y[cat][p.i, p.target]
    r = np.log(a / p.yhat.to_numpy())
    q = pd.qcut(meta["population"].fillna(meta["population"].median()).rank(method="first"), strata, labels=False).to_numpy()[p.i]
    rows = []
    for T0 in sorted(p.origin.unique()):
        for h in sorted(p.h.unique()):
            cur = ((p.origin == T0) & (p.h == h)).to_numpy()
            past = ((p.target <= T0) & (p.h == h)).to_numpy()
            if past.sum() < 500 or not cur.any():
                continue
            for s_ in range(strata):
                c, ps = cur & (q == s_), past & (q == s_)
                row = {"origin": T0, "h": h, "n": int(c.sum())}
                wis, f, y = 0.5 * np.abs(a[c] - p.yhat.to_numpy()[c]), p.yhat.to_numpy()[c], a[c]
                for lv in levels:
                    lo, hi = np.quantile(r[ps], [(1 - lv) / 2, (1 + lv) / 2])
                    L, U = f * np.exp(lo), f * np.exp(hi)
                    al = 1 - lv
                    row[f"покрытие {int(lv * 100)}%"] = float(((y >= L) & (y <= U)).mean())
                    row[f"ширина {int(lv * 100)}%, %"] = float(100 * np.mean((U - L) / f))
                    wis = wis + (al / 2) * ((U - L) + 2 / al * (L - y) * (y < L) + 2 / al * (y - U) * (y > U))
                row["WIS"] = float(np.mean(wis / (len(levels) + 0.5)))
                rows.append(row)
    d = pd.DataFrame(rows)
    cols = [k for k in d.columns if k not in ("origin", "h", "n")]
    return d.groupby("h").apply(lambda g: pd.Series({k: np.average(g[k], weights=g.n) for k in cols}))
