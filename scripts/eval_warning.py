"""Оценка системы раннего предупреждения: точность и подъём над базовой частотой тех же МО, полнота, опережение
в днях; модель подавления (логистическая регрессия, проверка по месяцам — модель не видит месяц, который оценивает);
абляции источников (акты, новости МЧС) и плацебо со сдвигом дат. Запуск: python scripts/eval_warning.py [llm|regex]"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ews import panel as pnl, text_features as tf, warning as w
from ews.paths import OUTPUTS

rng = np.random.default_rng(42)
TARGET = "провал Маркетплейсы"          # категория выбрана заранее, по опыту «доза — ответ»; остальные — в таблице


def features(ep, meta):
    X = pd.DataFrame({"log новостей": np.log1p(ep.n_news), "есть акт": (ep.n_acts > 0).astype(float),
                      "акт по перечню МО": ep.local_act.astype(float), "акт на весь субъект": ep.region_act.astype(float),
                      "log пострадавших": np.log1p(ep.people),
                      "log население": np.log(meta.population.reindex(ep.territory_id).fillna(meta.population.median()).to_numpy())})
    for c in tf.RELEVANT:
        X[c] = (ep.cause == c).astype(float).to_numpy()
    return X


def suppression(ep, X, y, online=False):
    """Вероятность провала для каждого эпизода. По умолчанию модель обучена на эпизодах других месяцев (перекрёстная
    проверка по месяцам); online=True — только на эпизодах, исход которых уже известен (месяц начала + 1 < m).
    Признаки эпизода собраны по всем его тревогам, включая пришедшие после начала, поэтому и онлайн-вариант — оценка
    сверху."""
    p = np.full(len(ep), np.nan)
    for m in ep.m.unique():
        tr = ((ep.m + 1) < m).to_numpy() if online else (ep.m != m).to_numpy()
        te = (ep.m == m).to_numpy()
        if y[tr].sum() < 5:
            continue
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def summarize(ep, B, name, region):
    """Точность эпизодов и подъём над базой тех же МО: база эпизода — частота провала в его МО по всем месяцам окна
    (провал в месяце m или m + 1), так что состав МО и регионов на подъём не влияет."""
    row = {"вариант": name, "эпизодов": len(ep)}
    for lbl in list(w.CATS.values()) + ["любой"]:
        prec = ep[f"провал {lbl}"].mean()
        row[f"{lbl}: точность"] = prec
        row[f"{lbl}: подъём"] = prec / B[lbl][ep.i].mean()
    lead = ep.loc[ep[TARGET], "опережение Маркетплейсы"]
    row["опережение, дней (медиана)"] = lead.median()
    if len(ep):                                            # эпизоды одного акта на весь регион не независимы:
        key = pd.Series(list(zip(region[ep.i], ep.m)), index=ep.index)   # единица бутстрепа — регион × месяц
        g = ep.assign(k=key, b=B["Маркетплейсы"][ep.i]).groupby("k").agg(hits=(TARGET, "sum"), exp=("b", "sum"))
        boot = np.random.default_rng(0)
        idx = boot.integers(0, len(g), (2000, len(g)))
        bs = g["hits"].to_numpy()[idx].sum(1) / g["exp"].to_numpy()[idx].sum(1)
        row["кластеров регион × месяц"] = len(g)
        row["Маркетплейсы: подъём, 2,5%"], row["Маркетплейсы: подъём, 97,5%"] = np.percentile(bs, [2.5, 97.5])
    return row


def main(source="llm"):
    Y, ids, months, meta = pnl.load()
    region = meta["region_code"].astype(int).to_numpy()
    Z, dips = w.gsr()
    al = w.alerts(source)
    ep = w.match(w.episodes(al), dips, months, ids)
    lo, hi = months.index("2024-02"), months.index("2024-11")
    ep = ep[(ep.m >= lo) & (ep.m <= hi)].reset_index(drop=True)
    B = w.mo_base_rate(dips, lo, hi)
    res = [{"вариант": "базовая частота (все МО)", **{f"{k}: точность": v.mean() for k, v in B.items()}}]
    res.append(summarize(ep, B, "все эпизоды", region))
    res.append(summarize(ep[ep.n_acts > 0], B, "только с актом", region))
    res.append(summarize(ep[ep.n_news > 0], B, "только с новостями МЧС", region))
    res.append(summarize(ep[(ep.n_news > 0) & (ep.n_acts > 0)], B, "акт и новости", region))
    res.append(summarize(ep[ep.local_act | (ep.n_news > 0)], B, "адресные (акт по перечню или новость с МО)", region))
    X = features(ep, meta)
    y = ep[TARGET].to_numpy()
    p = suppression(ep, X.to_numpy(), y)
    ok = ~np.isnan(p)
    auc = roc_auc_score(y[ok], p[ok]) if ok.sum() > 20 and 0 < y[ok].mean() < 1 else np.nan
    for q, nm in [(0.5, "модель подавления: верхние 50% (полнота)"), (0.8, "модель подавления: верхние 20% (точность)")]:
        thr = np.nanquantile(p, q)
        res.append(summarize(ep[ok & (p >= thr)], B, nm, region))
    po = suppression(ep, X.to_numpy(), y, online=True)             # онлайн: обучение только на прошлых исходах
    ok2 = ~np.isnan(po)
    auc_online = roc_auc_score(y[ok2], po[ok2]) if ok2.sum() > 20 and 0 < y[ok2].mean() < 1 else np.nan
    if ok2.sum() > 20:
        thr = np.nanquantile(po, 0.5)                               # порог — медиана онлайн-прогнозов за всё окно
        res.append(summarize(ep[ok2 & (po >= thr)], B, f"онлайн-модель подавления: верхние 50% (эпизодов с прогнозом {int(ok2.sum())})", region))
    # плацебо: те же эпизоды, начало сдвинуто на ±3–6 месяцев с переносом по кругу внутри окна 02–11.2024
    pl = ep.copy()
    shift = rng.choice([-6, -5, -4, -3, 3, 4, 5, 6], len(pl))
    target = [pd.Period(months[lo + (m - lo + k) % (hi - lo + 1)], "M") for m, k in zip(pl.m, shift)]
    pl["start"] = [t.start_time + pd.Timedelta(days=min(st.day, t.days_in_month) - 1) for t, st in zip(target, pl.start)]
    pl = w.match(pl.drop(columns=[c for c in pl.columns if c.startswith(("провал", "опережение"))] + ["i", "m"]), dips, months, ids)
    res.append(summarize(pl, B, "плацебо: даты сдвинуты на 3–6 мес.", region))
    res.append(summarize(pl[pl.n_acts > 0], B, "плацебо: только с актом", region))       # пара к строке «только с актом»
    r = pd.DataFrame(res)
    pd.set_option("display.width", 260)
    print(f"источник новостей: {source}; эпизодов 2024 г.: {len(ep)}; AUC модели подавления ({TARGET}): {auc:.3f}, онлайн: {auc_online:.3f}")
    print(r.round(3).to_string(index=False))
    r.to_csv(OUTPUTS / "tables" / f"warning_{source}.csv", index=False)
    pd.DataFrame([{"модель подавления": "перекрёстная проверка по месяцам", "AUC": auc},
                  {"модель подавления": "онлайн", "AUC": auc_online}]).to_csv(OUTPUTS / "tables" / f"warning_auc_{source}.csv", index=False)
    ep.assign(p=p).to_csv(OUTPUTS / f"warning_episodes_{source}.csv", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
