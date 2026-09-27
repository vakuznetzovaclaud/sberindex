"""Разбор реального случая: паводок весны 2024 г. (Оренбургская, Курганская, Тюменская области). Хронология текста
(первые сообщения о подтоплении, акты о ЧС) против месяца, когда траты за него становятся известны; отклик трат МО
по категориям; доза — ответ: число сообщений о реальном подтоплении с названием МО до конца апреля против отклика.
Источники сообщений: новости сайтов ГУ МЧС (заголовок + текст) и публичные Telegram-каналы МЧС и региональных новостей;
разметка «реальное событие» — та же модель и те же правила. Запуск: python scripts/case_flood2024.py [модель остатков]"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from ews import evaluate as ev, news_classify as nc, news_telegram as nt, panel as pnl
from ews.geo import Matcher
from ews.paths import INTERIM, OUTPUTS, PROCESSED

REGIONS = {56: "Оренбургская обл.", 45: "Курганская обл.", 72: "Тюменская обл."}
CATS = pnl.CATS_SHORT
FLOOD = nc.FLOOD_WORDS
WINDOW = ("2024-03-15", "2024-04-30")
NEWS_CHANNELS = {"orenburg_online56", "kurganskayaobl"}        # региональные новостные ленты; остальные каналы — МЧС


def messages():
    """Сообщения о реальном подтоплении в окне: (источник, регион, дата, текст для привязки)."""
    rows = []
    lab = nc.labels()
    lab = lab[lab.region_code.isin(REGIONS) & lab.dt.between(WINDOW[0], WINDOW[1] + "T23:59")]
    for r in lab[lab.apply(nc.is_event, axis=1)].itertuples():
        b = nc.body_path(r.region_code, r.id)
        text = r.title + " ; " + " ; ".join(r.места or []) + (" ; " + b.read_text(encoding="utf-8") if b.exists() else "")
        if pd.Series([text]).str.contains(FLOOD, case=False, regex=True).iloc[0]:
            rows.append(("сайт МЧС", r.region_code, r.dt[:10], text))
    f = INTERIM / "tg_flood_labels.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"нет {f}: разметку Telegram делает make texts")
    posts = {p["post"]: p for p in nt.flood_posts()}
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if nc.is_event(r) and r["post"] in posts and WINDOW[0] <= r["dt"][:10] <= WINDOW[1]:
            p = posts[r["post"]]
            kind = "Telegram: новости" if r["post"].split("/")[0] in NEWS_CHANNELS else "Telegram: МЧС"
            rows.append((kind, r["region_code"], r["dt"][:10], p["text"] + " ; " + " ; ".join(r.get("места") or [])))
    return pd.DataFrame(rows, columns=["источник", "region_code", "date", "text"])


def main(model="lgbm"):
    Y, ids, months, meta = pnl.load()
    apr, may = months.index("2024-04"), months.index("2024-05")
    R = {lbl: ev.h1_response(model, c) for c, lbl in CATS.items()}
    evt = pd.read_parquet(PROCESSED / "events.parquet")
    evt = evt[evt.region_code.isin(REGIONS) & (evt.cause == "паводок/наводнение") & ~evt.old & evt.pub_date.between("2024-03-01", "2024-05-31")]
    msg = messages()
    M = Matcher()
    msg["mo"] = [M.match(t, rc) for t, rc in zip(msg.text, msg.region_code)]
    timeline, rows = [], []
    for rc, rname in REGIONS.items():
        m = msg[msg.region_code == rc]
        acts = evt[evt.region_code == rc].drop_duplicates("eo")
        timeline.append({"регион": rname, "первое сообщение: сайт МЧС": m[m["источник"] == "сайт МЧС"].date.min(),
                         "первое сообщение: Telegram": m[m["источник"].str.startswith("Telegram")].date.min(),
                         "первое сообщение: Telegram МЧС": m[m["источник"] == "Telegram: МЧС"].date.min(),
                         "сообщений о подтоплении": len(m), "акт о ЧС подписан": acts.doc_date.min(),
                         "акт опубликован": acts.pub_date.min(), "траты за апрель — не раньше": "2024-04-30"})
        in_acts = set(evt[(evt.region_code == rc) & evt.territory_id.notna() & (evt.scope == "перечень")].territory_id.astype(int))
        for k, tid in enumerate(ids):
            if int(meta.region_code.iloc[k]) != rc:
                continue
            hit = m[m.mo.apply(lambda s: tid in s)]
            rows.append({"регион": rname, "МО": meta.name.iloc[k], "territory_id": tid, "сообщений": len(hit),
                         "сайт МЧС": int((hit["источник"] == "сайт МЧС").sum()), "Telegram": int(hit["источник"].str.startswith("Telegram").sum()),
                         "Telegram МЧС": int((hit["источник"] == "Telegram: МЧС").sum()), "Telegram: новости": int((hit["источник"] == "Telegram: новости").sum()),
                         "в акте по перечню": tid in in_acts, "первое сообщение": hit.date.min() if len(hit) else None,
                         "первое сообщение МЧС": hit[hit["источник"] != "Telegram: новости"].date.min() if (hit["источник"] != "Telegram: новости").any() else None,
                         "первое сообщение на сайте МЧС": hit[hit["источник"] == "сайт МЧС"].date.min() if (hit["источник"] == "сайт МЧС").any() else None,
                         **{f"{lbl} апрель, %": R[lbl][k, apr] for lbl in CATS.values()},
                         **{f"{lbl} май, %": R[lbl][k, may] for lbl in CATS.values()}})
    tl, d = pd.DataFrame(timeline), pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    print(tl.to_string(index=False))
    d.to_csv(OUTPUTS / "case_flood2024.csv", index=False)
    tl.to_csv(OUTPUTS / "case_flood2024_timeline.csv", index=False)
    print(f"\nМО трёх регионов: {len(d)}; с сообщениями о подтоплении: {(d['сообщений'] > 0).sum()} "
          f"(сайт МЧС — {(d['сайт МЧС'] > 0).sum()}, Telegram — {(d['Telegram'] > 0).sum()})")
    grp = np.where(d["сообщений"] >= 5, "5+ сообщений", np.where(d["сообщений"] > 0, "1–4 сообщения", "без сообщений"))
    cols = [c for c in d.columns if c.endswith(", %")]
    print(d.groupby(grp)[cols].median().round(2).T.to_string())
    hit = d[d["сообщений"] > 0]
    for lbl in CATS.values():
        rho, p = stats.spearmanr(hit["сообщений"], hit[f"{lbl} апрель, %"], nan_policy="omit")
        u = stats.mannwhitneyu(d.loc[grp == "5+ сообщений", f"{lbl} апрель, %"].dropna(),
                               d.loc[grp == "без сообщений", f"{lbl} апрель, %"].dropna(), alternative="less").pvalue
        print(f"  {lbl:15s} апрель: доза — ответ ρ = {rho:+.2f} (p = {p:.3f}, n = {len(hit)}); 5+ против без сообщений p = {u:.3f}")
    # абляция источников: та же доза — ответ по каждому набору источников
    abl = []
    for name, src in [("сайты МЧС", {"сайт МЧС"}), ("Telegram МЧС", {"Telegram: МЧС"}), ("сайты + Telegram МЧС", {"сайт МЧС", "Telegram: МЧС"}),
                      ("Telegram: новостные ленты", {"Telegram: новости"}), ("все источники", {"сайт МЧС", "Telegram: МЧС", "Telegram: новости"})]:
        sub = msg[msg["источник"].isin(src)]
        dose = np.array([sum(tid in s_ for s_ in sub.mo) for tid in d.territory_id])
        row = {"источники": name, "сообщений": len(sub), "МО с сообщениями": int((dose > 0).sum())}
        for lbl in ["Все", "Маркетплейсы"]:
            y = d[f"{lbl} апрель, %"].to_numpy()
            m = dose > 0
            rho, p = stats.spearmanr(dose[m], y[m], nan_policy="omit") if m.sum() >= 5 else (np.nan, np.nan)
            row[f"{lbl}: ρ"], row[f"{lbl}: p"] = rho, p
        abl.append(row)
    abl = pd.DataFrame(abl)
    abl.to_csv(OUTPUTS / "case_flood2024_sources.csv", index=False)
    print("\nабляция источников (доза — ответ по МО с сообщениями, апрель):")
    print(abl.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
