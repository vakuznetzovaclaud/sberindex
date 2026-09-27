"""Система раннего предупреждения в духе EMBERS (Ramakrishnan et al., KDD 2014): текстовые тревоги по МО собираются в
эпизоды, модель подавления оценивает вероятность, что эпизод обернётся провалом трат, и эпизоды сверяются с эталоном
провалов (GSR), который строится по данным трат независимо от текста. Метрики — точность, подъём над базовой
частотой тех же МО, полнота, опережение."""
import numpy as np
import pandas as pd

from . import acts, evaluate as ev, panel as pnl, text_features as tf
from .paths import PROCESSED

CATS = {c: pnl.CATS_SHORT[c] for c in (0, 3, 4, 5)}


def alerts(source="llm"):
    """Тревоги с датой знания: акты о введении или изменении режима (дата публикации) и новости МЧС о реальных
    событиях с упоминанием МО (дата новости)."""
    _, ids, _, _ = pnl.load()
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ev = ev[ev.territory_id.notna() & ev.action.isin(["введение", "изменение"]) & ~ev.old & ev.cause.isin(tf.RELEVANT)]
    a = pd.DataFrame({"territory_id": ev.territory_id.astype(int), "date": pd.to_datetime(ev.pub_date), "source": "акт",
                      "local": (ev.scope == "перечень").to_numpy(), "cause": ev.cause, "people": ev.people.fillna(0).astype(float),
                      "ref": ev.eo.astype(str), "text": ev.cause_text.fillna("")})
    n = tf.news_links(source=source)
    n = n[n.territory_id >= 0]
    b = pd.DataFrame({"territory_id": n.territory_id.astype(int), "date": pd.to_datetime(n.dt.str[:10]), "source": "МЧС",
                      "local": True, "cause": n.title.map(acts.cause_category), "people": 0.0, "ref": n.id.astype(str),
                      "text": n.title})
    al = pd.concat([a, b], ignore_index=True)
    return al[al.territory_id.isin(set(ids))].sort_values(["territory_id", "date"]).reset_index(drop=True)


def episodes(al, gap_days=21):
    """Тревоги одного МО, между которыми не больше gap_days дней, сливаются в эпизод (начало — первая тревога)."""
    out = []
    for tid, g in al.groupby("territory_id"):
        cur, last = [], None
        for r in g.itertuples():
            if cur and (r.date - last).days > gap_days:
                out.append(_close(tid, cur)); cur = []
            cur.append(r); last = r.date
        out.append(_close(tid, cur))
    return pd.DataFrame(out)


def _close(tid, rs):
    act = [r for r in rs if r.source == "акт"]
    news = [r for r in rs if r.source == "МЧС"]
    causes = pd.Series([r.cause for r in rs])
    return {"territory_id": tid, "start": min(r.date for r in rs), "end": max(r.date for r in rs),
            "n_news": len({r.ref for r in news}), "n_acts": len({r.ref for r in act}),
            "local_act": any(r.local for r in act), "region_act": any(not r.local for r in act),
            "people": max([r.people for r in act] + [0.0]), "cause": causes.mode().iloc[0],
            "first_source": rs[0].source, "texts": " | ".join(dict.fromkeys(r.text for r in rs if r.text))[:300]}


def gsr(model="lgbm", thr=-2.0):
    """Эталон провалов Z[кат, МО, месяц]: остаток h = 1 в единицах шума МО минус медиана месяца по МО (как у детекторов);
    провал — ниже thr. Строится только по тратам, текст в нём не участвует."""
    Z = {lbl: ev.h1_response(model, c, scale=True) for c, lbl in CATS.items()}
    return Z, {lbl: Z[lbl] <= thr for lbl in Z}


def match(ep, dips, months, ids, lag=1):
    """Эпизод попал, если провал в месяце начала эпизода или в следующем (lag); опережение — от начала эпизода до конца
    месяца провала (раньше конца месяца траты за него не посчитать)."""
    pos = {t: k for k, t in enumerate(ids)}
    mi = {m: k for k, m in enumerate(months)}
    ep = ep.copy()
    ep["i"] = ep.territory_id.map(pos)
    ep["m"] = ep.start.dt.strftime("%Y-%m").map(mi)
    ep = ep.dropna(subset=["i", "m"]).astype({"i": int, "m": int})
    for lbl, D in dips.items():
        hit, lead = [], []
        for r in ep.itertuples():
            ts = [t for t in range(r.m, min(r.m + lag + 1, D.shape[1])) if D[r.i, t]]
            hit.append(bool(ts))
            lead.append((pd.Period(months[ts[0]], "M").end_time.normalize() - r.start).days if ts else np.nan)
        ep[f"провал {lbl}"] = hit
        ep[f"опережение {lbl}"] = lead
    ep["провал любой"] = ep[[f"провал {l}" for l in dips]].any(axis=1)
    return ep


def mo_base_rate(dips, lo, hi, lag=1):
    """Базовая частота по каждому МО: доля месяцев m окна lo..hi, где провал случился в месяце m или m + lag."""
    anyd = np.logical_or.reduce(list(dips.values()))
    return {lbl: np.array([D[:, m:m + lag + 1].any(1) for m in range(lo, hi + 1)]).mean(0)
            for lbl, D in list(dips.items()) + [("любой", anyd)]}
