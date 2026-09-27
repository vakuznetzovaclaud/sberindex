"""Главный эксперимент: сдвигают ли события из официальных актов траты МО и предсказывает ли число новостей о
последствиях глубину сдвига. Запуск: python scripts/dose_response.py [модель остатков, по умолчанию lgbm] [z — отклик в единицах шума МО]

Отклик — остаток одношагового прогноза (h = 1, log факт − log прогноз) за вычетом медианы по всем МО того же месяца
(общая ошибка месяца — не эффект события), в % : R0 — месяц события, R1 — следующий, R = (R0 + R1) / 2.
Единица вывода — акт (его МО коррелированы). Эффект акта — средний R его МО в месяц события минус средний R тех же МО в
другие месяцы окна (плацебо-даты, не ближе двух месяцев к событию): так сравниваются одни и те же МО, и постоянные
особенности МО эффектом не считаются. Проверка — критерий знаковых рангов Уилкоксона по актам; p с поправкой
Бонферрони — на пять категорий одной группы актов и на все проверки таблицы.
Доза — ответ: внутри акта (МО одного события), ранговая корреляция числа новостей МЧС о последствиях с упоминанием МО
и отклика МО, после вычитания средних по акту."""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from ews import evaluate as ev, news_intensity as ni, panel as pnl, text_features as tf
from ews.geo import Matcher
from ews.paths import OUTPUTS, PROCESSED

CATS = pnl.CATS_SHORT                     # категории разбора событий: индекс в панели -> подпись
rng = np.random.default_rng(42)


def act_table(months, ids):
    """Акты 2024 г. о введении или изменении режима с установленной причиной: месяц события и индексы МО панели."""
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[ev.territory_id.notna() & ev.action.isin(["введение", "изменение"]) & ~ev.old & ev.cause.isin(tf.RELEVANT + ["иное/не указана"])]
    ev["start"] = np.minimum(pd.to_datetime(ev.doc_date), pd.to_datetime(ev.pub_date)).dt.strftime("%Y-%m")
    ev = ev[(ev.start >= "2024-02") & (ev.start <= "2024-11")]
    pos = {t: k for k, t in enumerate(ids)}
    out = []
    for eo, g in ev.groupby("eo"):
        mo = sorted({pos[int(t)] for t in g.territory_id if int(t) in pos})
        if mo:
            r = g.iloc[0]
            out.append({"eo": eo, "cause": r.cause, "scope": r.scope, "regime": r.regime, "region_code": r.region_code,
                        "m": months.index(r.start), "mo": mo, "doc_date": r.doc_date, "pub_date": r.pub_date})
    return pd.DataFrame(out)


def act_effect(R, mo, m, window):
    """Средний отклик МО акта в месяц события и в плацебо-месяцы окна (|m' − m| ≥ 2); возвращает (факт, плацебо[])."""
    resp = lambda t: np.nanmean([np.nanmean([R[i, t], R[i, t + 1]]) for i in mo])
    fact = resp(m)
    plac = [resp(t) for t in window if abs(t - m) >= 2]
    return fact, np.array(plac)


def main(model="lgbm", scale=""):
    Y, ids, months, meta = pnl.load()
    R = {lbl: ev.h1_response(model, c, scale == "z") for c, lbl in CATS.items()}
    window = [months.index(m) for m in months if "2024-02" <= m <= "2024-11"]
    acts = act_table(months, ids)
    rows = []
    for a in acts.itertuples():
        row = {"eo": a.eo, "cause": a.cause, "scope": a.scope, "regime": a.regime, "m": months[a.m], "n_mo": len(a.mo)}
        for lbl in CATS.values():
            fact, plac = act_effect(R[lbl], a.mo, a.m, window)
            row[f"эффект {lbl}"] = fact - np.nanmean(plac)
            row[f"факт {lbl}"] = fact
        rows.append(row)
    d = pd.DataFrame(rows)
    sfx = "_z" if scale == "z" else ""
    d.to_csv(OUTPUTS / f"dose_response_acts{sfx}.csv", index=False)

    summ = []
    groups = [("акты с установленной причиной", d[d.cause != "иное/не указана"]), ("причина не указана", d[d.cause == "иное/не указана"])]
    groups += [(c, g) for c, g in d[d.cause != "иное/не указана"].groupby("cause")]
    groups += [(f"охват: {s}", g) for s, g in d[d.cause != "иное/не указана"].groupby("scope")]
    groups += [(f"режим: {s}", g) for s, g in d[d.cause != "иное/не указана"].groupby("regime")]
    for name, g in groups:
        if len(g) < 3:
            continue
        row = {"группа": name, "актов": len(g), "МО-событий": int(g.n_mo.sum())}
        for lbl in CATS.values():
            x = g[f"эффект {lbl}"].dropna()
            row[f"{lbl}: медиана"] = x.median()
            row[f"{lbl}: p"] = stats.wilcoxon(x, alternative="less").pvalue if len(x) >= 5 else np.nan
        summ.append(row)
    # акты по перечню МО: названные МО против остальных МО того же региона (снимает и общие для региона сдвиги)
    reg = meta.region_code.astype(int).to_numpy()
    for a in acts[(acts.scope == "перечень") & (acts.cause != "иное/не указана")].itertuples():
        others = [k for k in np.where(reg == a.region_code)[0] if k not in set(a.mo)]
        if len(others) < 3:
            continue
        for lbl in CATS.values():
            f1, p1 = act_effect(R[lbl], a.mo, a.m, window)
            f0, p0 = act_effect(R[lbl], others, a.m, window)
            d.loc[d.eo == a.eo, f"внутри региона {lbl}"] = (f1 - np.nanmean(p1)) - (f0 - np.nanmean(p0))
    w_reg = d.dropna(subset=[f"внутри региона {CATS[0]}"]) if f"внутри региона {CATS[0]}" in d else d.iloc[:0]
    if len(w_reg) >= 5:
        row = {"группа": "перечень МО против остальных МО региона", "актов": len(w_reg), "МО-событий": int(w_reg.n_mo.sum())}
        for lbl in CATS.values():
            x = w_reg[f"внутри региона {lbl}"].dropna()
            row[f"{lbl}: медиана"] = x.median()
            row[f"{lbl}: p"] = stats.wilcoxon(x, alternative="less").pvalue if len(x) >= 5 else np.nan
        summ.append(row)
    s = pd.DataFrame(summ)
    # поправка Бонферрони: семейство — пять категорий одной группы актов; вторая — на все p таблицы
    family = len(CATS)
    n_all = int(s[[f"{lbl}: p" for lbl in CATS.values()]].notna().sum().sum())
    for lbl in CATS.values():
        s.insert(s.columns.get_loc(f"{lbl}: p") + 1, f"{lbl}: p с поправкой", (family * s[f"{lbl}: p"]).clip(upper=1))
    for lbl in CATS.values():
        s[f"{lbl}: p с поправкой на все проверки"] = (n_all * s[f"{lbl}: p"]).clip(upper=1)
    s.to_csv(OUTPUTS / f"dose_response_summary{sfx}.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"модель остатков: {model}; отклик в {'z (шум МО)' if scale == 'z' else '%'}; актов 2024 г. с МО панели: {len(d)}")
    print(s.round(3).to_string(index=False))

    # доза — ответ внутри актов: новости МЧС о последствиях с упоминанием МО в окне [−14; +30] дней от даты акта
    M = Matcher()
    cov = tf.covered_regions()
    pairs = []
    for a in acts[acts.cause != "иное/не указана"].itertuples():
        if a.region_code not in cov:
            continue
        start = pd.Timestamp(min(a.doc_date, a.pub_date))
        for i in a.mo:
            n, _ = ni.intensity(M, a.region_code, ids[i], a.cause, (start - pd.Timedelta(days=14)).strftime("%Y-%m-%d"),
                                (start + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
            pairs.append({"eo": a.eo, "cause": a.cause, "i": i, "news": n,
                          **{lbl: np.nanmean([R[lbl][i, a.m], R[lbl][i, a.m + 1]]) for lbl in CATS.values()}})
    p = pd.DataFrame(pairs)
    p.to_csv(OUTPUTS / f"dose_response_pairs{sfx}.csv", index=False)
    varied = p.groupby("eo").news.transform(lambda x: x.nunique() > 1)
    q = p[varied].copy()
    print(f"\nдоза — ответ: пар акт × МО в регионах с архивом МЧС {len(p)}, актов с разной дозой по МО {q.eo.nunique()}, пар {len(q)}")
    for lbl in CATS.values():
        dx = q.news - q.groupby("eo").news.transform("mean")
        dy = q[lbl] - q.groupby("eo")[lbl].transform("mean")
        ok = dy.notna()
        if ok.sum() < 10:
            continue
        rho, _ = stats.spearmanr(dx[ok], dy[ok])
        perm = []                                       # перестановка доз внутри акта
        for _ in range(2000):
            sh = q[ok].groupby("eo").news.transform(lambda x: rng.permutation(x.values))
            dxs = sh - q[ok].groupby("eo").news.transform("mean")
            perm.append(stats.spearmanr(dxs, dy[ok])[0])
        pv = (np.sum(np.array(perm) <= rho) + 1) / (len(perm) + 1)
        print(f"  {lbl:15s} ρ = {rho:+.3f}, p (перестановки внутри акта, односторонний) = {pv:.4f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
