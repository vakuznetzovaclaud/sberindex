"""Повторяется ли на других событиях 2024 г. то, что найдено на паводке: у МО, которые назвал официальный источник,
траты на маркетплейсах в месяц события ниже, чем у контроля. Событие — регион-месяц 2024-02…2024-11, в котором пост
официального канала (ГУ МЧС, губернатор или правительство, ЦУР) называет МО и говорит о последствиях ЧС; масштаб
события — наибольший масштаб поста (сводки дают числа нарастающим итогом). Тяжёлое событие — масштаб от 100 домов
или человек. Паводок весны 2024 г. (регионы 56, 45, 72, март–май) исключён: на нём сдвиг и найден. Правила отбора
постов, порог и тесты заданы до расчёта трат.
Исход — −z маркетплейсов (остатки прогноза в единицах шума МО, нормировка по прошлому, как у детекторов); вторично —
остаток ансамбля с горизонтом 2 (прогноз сделан до месяца события). Контроль — МО того же региона без упоминаний или
МО регионов без событий в этом месяце. Тест по событию — односторонний Манна–Уитни, итог — сумма z по Стауфферу
с весом √n. Для сравнения — те же тесты на сайтах ГУ МЧС (разметка работы) и паводок как положительный контроль.
Запуск: python scripts/eval_events_2024.py"""
import numpy as np
import pandas as pd
from scipy import stats

from ews import detect, evaluate as ev, news_official as no, panel as pnl, synth
from ews.paths import INTERIM, OUTPUTS, PROCESSED

MP, HEAVY = 4, 100
FLOOD = ([56, 45, 72], "2024-03", "2024-05")


def residual_h(Y, model, h):
    """Остаток прогноза с горизонтом h минус медиана месяца по всем МО; знак как у −z (больше — сильнее провал)."""
    p = ev.load(model, MP)
    p = p[p.h == h]
    r = np.full(Y[MP].shape, np.nan)
    r[p.i.to_numpy(), p.target.to_numpy()] = np.log(Y[MP][p.i, p.target]) - np.log(p.yhat.to_numpy())
    return -(r - np.nanmedian(r, 0, keepdims=True))


def events(links):
    """Регион-месяцы: названные МО и масштаб; links — строки (region_code, month, mo, масштаб)."""
    return links.groupby(["region_code", "month"]).agg(mo=("mo", lambda x: sorted(set(x))), масштаб=("масштаб", "max")).reset_index()


def stouffer(X, E, busy, ctrl, months, pos, reg):
    rows = []
    for _, e in E.iterrows():
        t, idx = months.index(e.month), [pos[m] for m in e.mo]
        c = ((reg == e.region_code) if ctrl == "тот же регион" else ~np.isin(reg, list(busy[e.month]))).copy()
        c[idx] = False
        v, ref = X[idx, t], X[c, t]
        v, ref = v[~np.isnan(v)], ref[~np.isnan(ref)]
        if len(v) == 0 or len(ref) < 3:
            continue
        p = float(np.clip(stats.mannwhitneyu(v, ref, alternative="greater").pvalue, 1e-12, 1 - 1e-12))
        rows.append((len(v), stats.norm.isf(p), float(np.median(v) - np.median(ref))))
    n, z, d = map(np.array, zip(*rows))
    zc = float((np.sqrt(n) * z).sum() / np.sqrt(n.sum()))
    return {"событий": len(n), "МО": int(n.sum()), "медиана разности −z": float(np.median(d)), "z Стауффера": zc, "p": float(stats.norm.sf(zc))}


def within_region(M, sel, X, months, pos, reg, lag=0):
    """Названные МО из sel против МО того же региона, не упомянутых в этом месяце ни в одном посте (M — все МО-месяцы
    с постами); итог — Стауффер с весом √n."""
    rows = []
    for (rc, mo), g in sel.groupby(["region_code", "month"]):
        t = months.index(mo) + lag
        if t >= X.shape[1]:
            continue
        c = (reg == rc).copy()
        c[[pos[m] for m in M[(M.region_code == rc) & (M.month == mo)].mo]] = False
        v, ref = X[[pos[m] for m in g.mo], t], X[c, t]
        v, ref = v[~np.isnan(v)], ref[~np.isnan(ref)]
        if len(v) == 0 or len(ref) < 3:
            continue
        p = float(np.clip(stats.mannwhitneyu(v, ref, alternative="greater").pvalue, 1e-12, 1 - 1e-12))
        rows.append((len(v), stats.norm.isf(p)))
    n, z = map(np.array, zip(*rows))
    return len(n), int(n.sum()), float((np.sqrt(n) * z).sum() / np.sqrt(n.sum()))


def variants(P, Z, H2, S, months, pos, reg):
    """Другие способы применить текст (заданы до расчёта): время события внутри месяца, устойчивость внимания,
    признание актом, текст как объяснение тревог детектора."""
    P = P.assign(day=P.dt.str[8:10].astype(int), date=P.dt.str[:10])
    M = P.groupby(["region_code", "month", "mo"]).agg(first=("day", "min"), days=("date", "nunique")).reset_index()
    acts = pd.read_parquet(PROCESSED / "events.parquet")
    acts = acts[(acts.action != "иное") & ~acts.old & acts.territory_id.notna()]
    listed = set(zip(acts.territory_id.astype(int), pd.to_datetime(acts.pub_date).dt.strftime("%Y-%m")))
    M["в акте"] = [(int(m), mo) in listed for m, mo in zip(M.mo, M.month)]
    early, late = within_region(M, M[M["first"] <= 15], Z, months, pos, reg), within_region(M, M[M["first"] > 15], H2, months, pos, reg, lag=1)
    parts = {"время: ранние — месяц события, поздние — следующий месяц по прогнозу до события": [early, late],
             "устойчивость: МО в постах 3 и более дня месяца": [within_region(M, M[M.days >= 3], Z, months, pos, reg)],
             "признание: МО в постах и в перечне акта того же месяца": [within_region(M, M[M["в акте"]], Z, months, pos, reg)]}
    rows = []
    for name, ps in parts.items():
        N = sum(q[1] for q in ps)
        zc = sum(np.sqrt(q[1]) * q[2] for q in ps) / np.sqrt(N)
        rows.append({"вариант": name, "событий": sum(q[0] for q in ps), "МО": N, "z Стауффера": zc, "p": stats.norm.sf(zc)})
    # тревоги детектора (сумма Стауффера, порог 3%) в МО-месяцах с постом и без: Мантель–Хензель по регион-месяцам
    lo, hi = months.index("2024-02"), months.index("2024-11")
    thr = np.nanquantile(S[:, lo:hi + 1], 0.97)
    has = np.zeros(S.shape, bool)
    has[[pos[m] for m in M.mo], [months.index(mo) for mo in M.month]] = True
    num = den = obs = exp = var = 0.0
    for t in range(lo, hi + 1):
        for r in np.unique(reg):
            k = (reg == r) & ~np.isnan(S[:, t])
            al, h = S[k, t] > thr, has[k, t]
            a, b, c, d = np.sum(al & h), np.sum(~al & h), np.sum(al & ~h), np.sum(~al & ~h)
            n = a + b + c + d
            if a + c == 0 or a + b == 0 or c + d == 0:
                continue
            num, den = num + a * d / n, den + b * c / n
            obs, exp = obs + a, exp + (a + b) * (a + c) / n
            var += (a + b) * (c + d) * (a + c) * (b + d) / (n ** 2 * (n - 1))
    rows.append({"вариант": "тревоги детектора при посте (отношение шансов Мантеля–Хензеля)", "событий": np.nan, "МО": np.nan,
                 "z Стауффера": num / den, "p": stats.chi2.sf((obs - exp) ** 2 / var, 1)})
    r = pd.DataFrame(rows).rename(columns={"z Стауффера": "z Стауффера или отношение шансов"})
    r["p с поправкой"] = (len(r) * r.p).clip(upper=1)
    return r


def main():
    Y, ids, months, meta = pnl.load()
    Z = -synth.h1_residuals(Y, months, synth.nat_series())[MP]
    H2 = residual_h(Y, "ens_main", 2)
    pos = {t: k for k, t in enumerate(ids)}
    reg = meta.loc[ids, "region_code"].astype(int).to_numpy()
    in_flood = lambda d: d.region_code.isin(FLOOD[0]) & d.month.between(FLOOD[1], FLOOD[2])

    P = no.event_posts()
    P = P[(P.month >= "2024-02") & (P.month <= "2024-11")].explode("mo")
    P = P[P.mo.isin(pos)]
    busy_tg = P.groupby("month").region_code.agg(set)
    S = pd.read_parquet(INTERIM / "mchs_event_links_llm.parquet").rename(columns={"territory_id": "mo"})
    S["month"], S["масштаб"] = S.dt.str[:7], 0
    S = S[(S.month >= "2024-02") & (S.month <= "2024-11") & S.mo.isin(pos)]
    busy_site = S.groupby("month").region_code.agg(set)

    E_tg, E_site = events(P[~in_flood(P)]), events(S[~in_flood(S)])
    heavy = E_tg[E_tg.масштаб >= HEAVY]
    M = P[P.kind == "mchs"]                               # каналы губернаторов пишут и о благоустройстве дворов — шумнее
    E_m = events(M[~in_flood(M)])
    print(f"официальные каналы: {P.channel.nunique()} каналов, {P.dt.nunique()} постов о событиях; событий без паводка {len(E_tg)}, "
          f"тяжёлых {len(heavy)}; сайты ГУ МЧС: событий без паводка {len(E_site)}")
    runs = [("официальные каналы", f"тяжёлые (масштаб ≥ {HEAVY}), основной тест", heavy, busy_tg, Z, "−z, месяц события"),
            ("официальные каналы", "все события", E_tg, busy_tg, Z, "−z, месяц события"),
            ("официальные каналы", f"тяжёлые (масштаб ≥ {HEAVY})", heavy, busy_tg, H2, "ансамбль, h = 2"),
            ("официальные каналы", "масштаб ≥ 1000", E_tg[E_tg.масштаб >= 1000], busy_tg, Z, "−z, месяц события"),
            ("только каналы ГУ МЧС", f"тяжёлые (масштаб ≥ {HEAVY})", E_m[E_m.масштаб >= HEAVY], M.groupby("month").region_code.agg(set), Z,
             "−z, месяц события"),
            ("сайты ГУ МЧС", "все события", E_site, busy_site, Z, "−z, месяц события"),
            ("официальные каналы", "паводок весны 2024 г. (контроль)", events(P[in_flood(P)]), busy_tg, Z, "−z, месяц события"),
            ("сайты ГУ МЧС", "паводок весны 2024 г. (контроль)", events(S[in_flood(S)]), busy_site, Z, "−z, месяц события")]
    rows = []
    for src, sel, E, busy, X, out in runs:
        for ctrl in ("тот же регион", "регионы без событий"):
            rows.append({"источник": src, "события": sel, "исход": out, "контроль": ctrl, **stouffer(X, E, busy, ctrl, months, pos, reg)})
    r = pd.DataFrame(rows)
    r["p с поправкой"] = np.where(r["события"].str.contains("основной"), (2 * r.p).clip(upper=1), np.nan)   # два контроля
    pd.set_option("display.width", 250)
    print(r.round(4).to_string(index=False))
    r.to_csv(OUTPUTS / "tables" / "events2024.csv", index=False)

    D = P[~in_flood(P)].groupby(["region_code", "month", "mo"]).agg(постов=("dt", "size"), масштаб=("масштаб", "max")).reset_index()
    D["z"] = [Z[pos[m], months.index(mo)] for m, mo in zip(D.mo, D.month)]
    D = D.dropna()
    dose = pd.DataFrame([{"мера": k, "ρ": stats.spearmanr(D[k], D.z)[0], "p": stats.spearmanr(D[k], D.z)[1], "МО-событий": len(D)}
                         for k in ("масштаб", "постов")])
    print("\nдоза — ответ по МО-событиям (−z маркетплейсов):")
    print(dose.round(4).to_string(index=False))
    dose.to_csv(OUTPUTS / "tables" / "events2024_dose.csv", index=False)

    v = variants(P[~in_flood(P)], Z, H2, detect.multicat_stouffer(synth.h1_residuals(Y, months, synth.nat_series())), months, pos, reg)
    print("\nдругие способы применить текст (внутри региона, −z маркетплейсов):")
    print(v.round(4).to_string(index=False))
    v.to_csv(OUTPUTS / "tables" / "events2024_variants.csv", index=False)


if __name__ == "__main__":
    main()
