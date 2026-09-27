"""Сборка лендинга: страница site/index.html из итоговых таблиц outputs/ (один файл без внешних зависимостей)
и презентация site/presentation.html.
Запуск: python scripts/build_site.py"""
import base64
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

from ews import evaluate as ev, panel as pnl, synth
from ews.paths import OUTPUTS, PROCESSED

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
# короткие имена: на телефоне подпись столбца — до ~20 знаков; * — вход без сезонности (материал 02)
NAMES = {"snaive": "сезонный наивный", "snaive_growth": "наивный + прирост г/г", "prophet_default": "Prophet",
         "prophet_timecast": "Prophet (TimeCast)", "prophet_tuned": "Prophet (подбор)",
         "panel_factor_panel": "панель без нац. ряда", "panel": "панельная модель", "lgbm": "LightGBM",
         "lgbm_global": "LightGBM, 6 категорий", "naive_sa": "наивный*", "ses_sa": "сглаживание*",
         "theta_sa": "Theta*", "timesfm25": "TimesFM-2.5", "tirex2": "TiRex-2", "chronos2": "Chronos-2",
         "timesfm25_sa": "TimesFM-2.5*", "tirex2_sa": "TiRex-2*", "chronos2_sa": "Chronos-2*",
         "chronos2_sa_cross": "Chronos-2* совместный", "chronos2_sa_cov": "Chronos-2* с ковариатами",
         "chronos2_sa_cov_cross": "Chronos-2* ковариаты + совм.", "ens_fm_sa": "ансамбль трёх FM*", "ens_main": "ансамбль",
         "lstm_sa": "LSTM*", "tcn_sa": "TCN*", "patchtst_sa": "PatchTST*"}
MON = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
mon = lambda ym: f"{MON[int(ym[5:7]) - 1]} {ym[2:4]}"          # «2024-04» -> «апр 24»
MON_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]
day_month = lambda d: f"{pd.Timestamp(d).day} {MON_GEN[pd.Timestamp(d).month - 1]}"   # «2024-03-28» -> «28 марта»
DET = {"6 категорий: GLR": "GLR, 6 категорий", "GLR (окно 3)": "GLR, общие траты", "6 категорий: Стауффер": "Стауффер, 6 категорий",
       "6 категорий: максимум": "максимум, 6 категорий", "ансамбль: Стауффер + GLR + ранг": "ансамбль-3",
       "пространственный скан": "скан по соседям"}
CHOSEN = "6 категорий: Стауффер"          # выбранный детектор (отчёт, раздел 7)
DOSE = {"акты с установленной причиной": "акты с установленной причиной", "причина не указана": "причина не указана",
        "лесные/природные пожары": "природные пожары", "метеоявления (ветер, снег, мороз)": "метеоявления",
        "паводок/наводнение": "паводок", "техногенная авария/разлив/обрушение/взрыв": "техногенная авария",
        "авария ЖКХ/тепло-, энерго-, водоснабжение": "авария ЖКХ", "охват: весь субъект": "акт на весь регион",
        "охват: перечень": "акт с перечнем МО", "режим: повышенная готовность": "повышенная готовность",
        "режим: чрезвычайная ситуация": "режим ЧС", "перечень МО против остальных МО региона": "МО из перечня vs регион"}
FAMILY = lambda m: ("ens" if m.startswith("ens") else "prophet" if m.startswith("prophet") else
                    "fm" if any(k in m for k in ("timesfm", "tirex", "chronos")) else
                    "nn" if any(k in m for k in ("lstm", "tcn", "patchtst")) else "stat")
# объяснения тревог недельного полигона; источник каждого — ссылка в словаре SRC
SRC = {"такси": "https://www.interfax.ru/russia/1075277",
       "бензин": "https://neftegaz.ru/news/petroleum-products/792449-stoimost-benzina-ai-95-vnov-obnovila-rekord-na-birzhe-rf/",
       "вузы": "https://www.kommersant.ru/doc/8708232",
       "концерты": "https://www.sostav.ru/publication/rossiyane-uvelichili-raskhody-na-razvlecheniya-v-nachale-2025-goda-73119.html",
       "кризис": "https://www.kommersant.ru/doc/7971552", "ндс": "https://volga.news/article/791561.html",
       "данные": "https://sberindex.ru/ru/dashboards/ver-izmenenie-trat-po-kategoriyam",
       "хостинг": "https://ruvds.com/ru/prices-2025/",
       "пересчёт": "https://sberindex.ru/api/files/download/b12466ff-c794-4342-9354-d612891e734a"}
STEP = "ступенька уровня: весь 2025 год г/г +10…+31%, в январе 2026 года ушла из базы; причина не установлена"   # «Универсальные магазины», 4 недели
_why = lambda t, k: f'{t} <a href="{SRC[k]}">источник</a>'
WEEKLY_WHY = {("2026-03-22", "Такси, каршеринг, аренда автомобилей"): _why("с 01.03.2026 действует закон о локализации такси: иномарки не включают в реестры", "такси"),
              ("2026-06-21", "Топливо"): _why("биржевой рекорд цены Аи-95, рост розничных цен в 82 регионах", "бензин"),
              ("2026-06-28", "Топливо"): _why("биржевой рекорд цены Аи-95, рост розничных цен в 82 регионах", "бензин"),
              ("2026-08-16", "Образование"): _why("платное обучение в вузах подорожало в среднем на 10,7%, по части направлений на 20–30%", "вузы"),
              ("2026-08-23", "Образование"): "то же: сезон оплаты обучения по новым ценам",
              ("2026-08-30", "Образование"): "то же: сезон оплаты обучения по новым ценам",
              ("2025-03-23", "Развлечения"): _why("начало 2025 года: спрос на концерты +40%, цены на культурные события +30%", "концерты"),
              ("2025-03-30", "Развлечения"): "то же",
              ("2025-04-06", "Развлечения"): "то же",
              ("2026-01-18", "Универсальные магазины"): _why("«эхо»: ровно год назад скачок +13% г/г ушёл из базы сравнения; плюс НДС с 01.01.2026", "ндс"),
              ("2026-01-18", "Контент и СМИ – Услуги"): _why("эффект базы: годом ранее, в неделю до 19.01.2025, траты категории резко упали (г/г с −34% до −46%) — в самих данных", "данные"),
              ("2025-01-19", "Компьютерные услуги"): _why("в декабре 2024 года хостинги и ИТ-сервисы перед НДС для УСН с 01.01.2025 (176-ФЗ) продлевали услуги по старой цене — декабрьский уровень был завышен, к середине января эффект прошёл", "хостинг"),
              ("2025-01-26", "Компьютерные услуги"): "то же",
              ("2026-05-24", "Хобби и увлечения"): _why("артефакт пересчёта СберИндекса: в выпусках эти недели сначала шли как +1,7…+3,1%, в выпуске 21.06 — +8…+9% по оценке Сбера, затем пересмотрены вниз", "пересчёт"),
              ("2026-05-31", "Хобби и увлечения"): "то же",
              ("2026-06-07", "Хобби и увлечения"): "то же",
              ("2025-01-19", "Универсальные магазины"): STEP,
              ("2025-01-26", "Универсальные магазины"): STEP,
              ("2025-02-16", "Универсальные магазины"): STEP,
              ("2025-03-16", "Универсальные магазины"): STEP,
              ("2026-09-20", "Топливо"): _why("эффект базы: в августе–сентябре 2025 года — топливный кризис (рекордные цены на бирже, дефицит на АЗС)", "кризис")}


def forecast_block():
    m = pd.read_csv(OUTPUTS / "tables" / "forecast_metrics.csv")
    m = m[m["МО"] == "все МО"]
    cats = [c for c in pnl.CATS if c in set(m["категория"])]
    wins = list(dict.fromkeys(m["окно"]))
    mae = {c: {w: [{"model": r.модель, "name": NAMES.get(r.модель, r.модель), "mae": float(r.MAE), "family": FAMILY(r.модель)}
                   for r in m[(m["категория"] == c) & (m["окно"] == w)].sort_values("MAE").itertuples()
                   if r.модель != "snaive"] for w in wins} for c in cats}
    t = m[m["окно"] == wins[0]]
    cols = [("MAE", "MAE, ₽", 0), ("MAPE, %", "MAPE, %", 2), ("WAPE, %", "WAPE, %", 2), ("MASE", "MASE", 3),
            ("R² г/г", "R² г/г", 3), ("R²_oos к наиву", "R²_oos", 3)]
    table = {c: [{"name": NAMES.get(r["модель"], r["модель"]), **{k: (None if pd.isna(r.get(k)) else float(r[k])) for k, _, _ in cols}}
                 for _, r in t[t["категория"] == c].sort_values("MAE").iterrows()] for c in cats}
    Y, ids, months, meta = pnl.load()
    series = []
    for n in ["panel", "lgbm", "tirex2", "tirex2_sa", "ens_main"]:
        if n in ev.available(0):
            p = ev.load(n, 0)
            e = pd.Series(np.abs(p.yhat.to_numpy() - Y[0][p.i, p.target])).groupby(p.target).mean()
            series.append({"name": NAMES[n], "values": [round(float(v)) for v in e.values], "bold": n == "ens_main"})
    labels = [mon(months[t]) for t in sorted(p.target.unique())]
    sub = pd.read_csv(OUTPUTS / "tables" / "forecast_metrics.csv")
    sub = sub[(sub["МО"] == "400 МО") & (sub["окно"] == "все точки")].pivot_table(index="категория", columns="модель", values="MAE")
    prophet = [{"cat": c, "default": float(sub.loc[c, "prophet_default"]), "timecast": float(sub.loc[c, "prophet_timecast"]),
                "tuned": float(sub.loc[c, "prophet_tuned"]), "ens": float(sub.loc[c, "ens_main"]),
                "best": "tuned" if sub.loc[c, "prophet_tuned"] < sub.loc[c, "ens_main"] else "ens"} for c in pnl.CATS if c in sub.index]
    key = [k for k in ["ens_main", "timesfm25_sa", "tirex2_sa", "chronos2_sa_cross", "chronos2_sa_cov_cross", "chronos2_sa",
                       "patchtst_sa", "lstm_sa", "lgbm", "panel", "panel_factor_panel", "snaive_growth", "prophet_default"]
           if k in set(t["модель"])]   # TCN (tcn_sa) в эту таблицу не входит: считался только для «Все категории», по остальным — NaN
    bycat = {}
    for met, best in (("MAE", min), ("MAPE, %", min), ("R² г/г", max)):
        pv = t.pivot_table(index="модель", columns="категория", values=met).reindex(index=key, columns=cats)
        bycat[met] = {"best": {c: best(pv[c].dropna()) for c in cats},
                      "rows": [{"name": NAMES.get(k, k), **{c: float(pv.loc[k, c]) for c in cats}} for k in key]}
    return {"categories": cats, "windows": wins, "mae": mae, "table": table, "prophet": prophet,
            "table_cols": [{"k": "name", "t": "Модель"}] + [{"k": k, "t": t, "n": 1, "d": d} for k, t, d in cols],
            "month": {"labels": labels, "series": series}, "bycat": bycat}


def mae_of(names, cat=0):
    names = [n for n in names if n in ev.available(cat)]
    return ev.table(names, cat).set_index("модель").MAE.to_dict() if names else {}


def fm_block():
    m = mae_of(["tirex2_raw", "tirex2", "tirex2_sa"])
    inputs = [{"label": "сырой ряд МО", "mae": m.get("tirex2_raw")}, {"label": "отклонение от фактора", "mae": m.get("tirex2")},
              {"label": "без сезонности", "mae": m.get("tirex2_sa"), "good": True}]
    Y, ids, months, meta = pnl.load()
    ref = ev.load("panel", 0)
    e_ref = pd.Series(np.abs(ref.yhat.to_numpy() - Y[0][ref.i, ref.target])).groupby(ref.origin).mean()
    series = []
    for n in ["timesfm25", "tirex2", "timesfm25_sa", "tirex2_sa"]:
        if n in ev.available(0):
            p = ev.load(n, 0)
            e = pd.Series(np.abs(p.yhat.to_numpy() - Y[0][p.i, p.target])).groupby(p.origin).mean()
            series.append({"name": NAMES[n], "values": [round(float(v), 3) for v in (e / e_ref).values]})
    return {"inputs": [r for r in inputs if r["mae"]],
            "context": {"labels": [mon(months[t]) for t in e_ref.index], "series": series}}


def detect_block():
    r = pd.read_csv(OUTPUTS / "detectors_synthetic.csv")
    cols = ["VUS-PR", "полнота@3%", "NAB-подобная", "F1 с допуском"]
    s = r.groupby("детектор")[cols].mean()
    fam = lambda d: "text" if "текст" in d else "sel" if d == CHOSEN else "ens" if "ансамбль" in d else "glr" if "GLR" in d else "base"
    rows = lambda t: [{"det": DET.get(d, d), "VUS-PR": round(float(v["VUS-PR"]), 3), "полнота@3%": round(float(v["полнота@3%"]), 3),
                       "NAB": round(float(v["NAB-подобная"]), 1), "F1": round(float(v["F1 с допуском"]), 3), "family": fam(d)}
                      for d, v in t.iterrows() if "текст" not in d]
    shapes = ["все формы"] + list(synth.SHAPES)                  # по формам: отраслевой шок не бьёт по всем категориям сразу
    synthetic = {sh: rows(s if sh == "все формы" else r[r["форма"] == sh].groupby("детектор")[cols].mean()) for sh in shapes}
    scen = {"текст слабый": "ловит 20%, ложных 9%", "текст хороший": "ловит 50%, ложных 5%", "текст идеальный": "ловит 80%, ложных 2%"}
    text = [{"label": "без текста", "vus": round(float(s.loc[CHOSEN, "VUS-PR"]), 3)}]
    text += [{"label": scen[d.replace("Стауффер + ", "")], "vus": round(float(s.loc[d, "VUS-PR"]), 3)} for d in s.index if "текст" in d]
    lv = ["0.5%", "1%", "2%", "3%", "5%", "10%"]              # кривая «задержка — ложные тревоги» (InDiD)
    cv = r[~r["детектор"].str.contains("текст")].groupby("детектор")[[f"{m}@{x}" for m in ("задержка с пропусками", "полнота") for x in lv]].mean()
    delay = cv[[f"задержка с пропусками@{x}" for x in lv]]
    # страница и слайды говорят, что у выбранного детектора наименьшая задержка при любой доле ложных тревог
    assert (delay.drop(CHOSEN) >= delay.loc[CHOSEN]).all().all(), "текст о кривой разошёлся с таблицей детекторов"
    lag = delay.loc[[d for d in delay.index if "ансамбль" in d]] - delay.loc[CHOSEN]   # отставание ансамблей от выбранного
    key = [CHOSEN, "ансамбль: Стауффер + GLR + ранг", "6 категорий: GLR", "поперечный ранг", "порог z", "BOCPD"]
    curve = {"levels": [x.replace(".", ",") for x in lv],
             "series": {m: [{"name": DET.get(d, d), "values": [round(float(cv.loc[d, f"{col}@{x}"]), 3) for x in lv]} for d in key]
                        for m, col in (("задержка, мес.", "задержка с пропусками"), ("полнота", "полнота"))}}
    b = pd.read_csv(OUTPUTS / "offline_breaks.csv")
    piv = b.pivot_table(index="месяц разладки", columns="категория", values="territory_id", aggfunc="count", fill_value=0)
    breaks = {"labels": [mon(m) for m in piv.index],
              "series": [{"name": c, "values": [int(v) for v in piv[c].values]} for c in pnl.CATS if c in piv]}
    w = pd.read_csv(OUTPUTS / "weekly_polygon.csv")
    w = w[(w["z без календаря"].abs() > 4) & ~w["календарная неделя"]]
    weekly = [{"week": pd.Timestamp(r["неделя"]).strftime("%d.%m.%Y"), "cat": r["категория"], "yoy": round(float(r["г/г"]), 1), "z": round(float(r["z без календаря"]), 1),
               "why": WEEKLY_WHY.get((r["неделя"], r["категория"]), "«эхо» прошлогоднего сдвига" if r.get("эхо") else "")}
              for _, r in w.sort_values("неделя").iterrows()]
    return {"metrics": ["VUS-PR", "полнота@3%", "NAB", "F1"], "shapes": shapes, "synthetic": synthetic, "text": text, "breaks": breaks, "curve": curve,
            "lag_lo": round(float(lag.min().min()), 2), "lag_hi": round(float(lag.max().max()), 2),
            "weekly": weekly}                                          # все тревоги по датам, без отбора


def prior_nums(ds):
    """Текстовый априор на полусинтетике: прибавка к выбранному детектору при слабом и хорошем тексте."""
    g = ds.groupby("детектор")[["VUS-PR", "задержка@3%"]].mean()
    weak, good = g.loc["Стауффер + текст слабый"], g.loc["Стауффер + текст хороший"]
    return {"pr_vus0": ru(g.loc[CHOSEN, "VUS-PR"], 3), "pr_gain_weak": ru(weak["VUS-PR"] - g.loc[CHOSEN, "VUS-PR"], 3),
            "pr_vus_good": ru(good["VUS-PR"], 3), "pr_d0": ru(g.loc[CHOSEN, "задержка@3%"], 2), "pr_d_good": ru(good["задержка@3%"], 2)}


def text_block():
    d = pd.read_csv(OUTPUTS / "dose_response_summary.csv")
    cats = ["Маркетплейсы", "Все", "Общепит", "Транспорт", "Продовольствие"]
    dose = [{"group": DOSE.get(r["группа"], r["группа"]), "n": int(r["актов"]), **{c: {"med": None if pd.isna(r[f"{c}: медиана"]) else round(float(r[f"{c}: медиана"]), 4),
                                                               "p": None if pd.isna(r[f"{c}: p"]) else round(float(r[f"{c}: p"]), 4)} for c in cats}}
            for _, r in d.iterrows() if r["актов"] >= 5]
    return {"categories": cats, "dose": dose}


def dmy(d):
    return pd.Timestamp(d).strftime("%d.%m.%Y")


def quote(t):
    """Дословная выдержка из акта или заголовка: в кавычках, обрезанное начало — с многоточием."""
    return "«" + ("…" if t[:1].islower() else "") + t + "»"


def feed_block(source="llm"):
    """Лента эпизодов 2024 г.: эпизоды с одним текстом и датой в одном регионе (акт на весь субъект) — одна карточка
    со списком МО. Оценку модели подавления не показываем: в режиме реального времени она немногим лучше случайной.
    Точки карты — центры МО."""
    Y, ids, months, meta = pnl.load()
    e = pd.read_csv(OUTPUTS / f"warning_episodes_{source}.csv")
    e["mo"] = [str(meta.loc[t, "name_short"]) for t in e.territory_id]
    e["region"] = [meta.loc[t, "region_name"] for t in e.territory_id]
    items = []
    e["texts"] = e["texts"].fillna("(текст акта не извлечён)")
    for (reg, start, end, texts), g in e.groupby(["region", "start", "end", "texts"], sort=False):
        hit = g["провал Маркетплейсы"].astype(bool)
        items.append({"region": reg, "dates": dmy(start) if start == end else f"{dmy(start)} — {dmy(end)}", "start": start, "cause": g.cause.iloc[0],
                      "texts": quote(str(texts)[:260]), "mos": g.mo.tolist()[:40], "n_mo": len(g), "n_hit": int(hit.sum()),
                      "hit_mos": g.loc[hit, "mo"].tolist()[:12], "acts": int(g.n_acts.max()), "news": int(g.n_news.max()),
                      "lead": None if g.loc[hit, "опережение Маркетплейсы"].dropna().empty else int(g.loc[hit, "опережение Маркетплейсы"].median()),
                      "hit": bool(hit.any())})
    items.sort(key=lambda x: x["start"])
    state = {}
    for _, r in e.iterrows():
        if state.get(r.territory_id) != "hit":
            state[r.territory_id] = "hit" if r["провал Маркетплейсы"] else "alert"
    pts = [{"lat": float(meta.loc[t, "lat"]), "lon": float(meta.loc[t, "lon"]), "state": state.get(t, "none"),
            "note": f"{meta.loc[t, 'name']}<br>{meta.loc[t, 'region_name']}"} for t in ids]
    return {"items": items, "map": pts, "regions": sorted({i["region"] for i in items}),
            "filters": ["с провалом после эпизода", "все эпизоды"]}


def case_block():
    tl = pd.read_csv(OUTPUTS / "case_flood2024_timeline.csv")
    d = pd.read_csv(OUTPUTS / "case_flood2024.csv")
    src = pd.read_csv(OUTPUTS / "case_flood2024_sources.csv")
    t0 = pd.to_datetime(tl[["первое сообщение: сайт МЧС", "акт опубликован"]].stack()).min()   # источники, которые собираются по всей стране
    day = lambda x: (pd.Timestamp(x) - t0).days if isinstance(x, str) else None
    timeline = [{"region": r["регион"], "first": day(r["первое сообщение: сайт МЧС"]), "act": day(r["акт опубликован"])} for _, r in tl.iterrows()]
    grp = np.where(d["сообщений"] >= 5, "5+ сообщ.", np.where(d["сообщений"] > 0, "1–4 сообщ.", "нет сообщ."))
    bars = [{"label": f"{mth}: {g}", "value": round(float(d.loc[grp == g, f"{c} {mth}, %"].median()), 2)}
            for c in ["Маркетплейсы"] for mth in ["апрель", "май"] for g in ["5+ сообщ.", "1–4 сообщ.", "нет сообщ."]]
    named = np.where(d["сайт МЧС"] > 0, "на сайтах МЧС", np.where(d["сообщений"] > 0, "другие источники", "без сообщений"))
    by_source = [{"label": f"{mth}: {g} ({(named == g).sum()})", "value": round(float(d.loc[named == g, f"Маркетплейсы {mth}, %"].median()), 2)}
                 for mth in ["апрель", "май"] for g in ["на сайтах МЧС", "другие источники", "без сообщений"]]
    sources = [{"src": r["источники"], "n": int(r["сообщений"]), "mo": int(r["МО с сообщениями"]), "all": float(r["Все: ρ"]), "all_p": float(r["Все: p"]),
                "mp": float(r["Маркетплейсы: ρ"]), "mp_p": float(r["Маркетплейсы: p"])} for _, r in src.iterrows()]
    return {"timeline": timeline, "bars": bars, "by_source": by_source, "sources": sources, "t0": day_month(t0), "end": (pd.Timestamp("2024-04-30") - t0).days}


def ru(v, d=0, sign=False):
    """Число по-русски: пробел в тысячах, запятая в дробях, минус — настоящий."""
    t = f"{v:,.{d}f}".replace(",", "\u00a0").replace(".", ",")   # неразрывный пробел: число не рвётся по строкам
    if sign and v > 0:
        t = "+" + t
    return t.replace("-", "−")


def plural(n, one, few, many):
    w = one if n % 10 == 1 and n % 100 != 11 else few if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else many
    return f"{n} {w}"


known = lambda r: bool(r["why"]) and "не установлена" not in r["why"]


def episodes(rows, gap=21):
    """Число эпизодов: тревоги одной категории, между которыми не больше gap дней, — один эпизод."""
    n, last = 0, {}
    for r in sorted(rows, key=lambda r: pd.to_datetime(r["week"], dayfirst=True)):
        d = pd.to_datetime(r["week"], dayfirst=True)
        if r["cat"] not in last or (d - last[r["cat"]]).days > gap:
            n += 1
        last[r["cat"]] = d
    return n


def extra_forecast(ens):
    """Проверки заголовочных чисел прогноза и ошибка по месяцам цели — из таблиц scripts/eval_forecasts.py;
    ens — MAE ансамбля по всем точкам."""
    ch = pd.read_csv(OUTPUTS / "tables" / "forecast_checks.csv").set_index("показатель")["значение"]
    bm = pd.read_csv(OUTPUTS / "tables" / "forecast_by_month.csv").set_index("месяц цели")
    lose = bm.index[bm["ens_main"] > bm["prophet_default"]]            # месяцы, где ансамбль хуже Prophet
    lose_text = (f"единственный из {len(bm)} месяцев, где ансамбль уступает" if len(lose) == 1 else
                 f"ансамбль уступает Prophet в {len(lose)} месяцах из {len(bm)}")
    return {"pro_dec": ru(bm.loc["2024-12", "prophet_default"]), "dec_raw": ru(bm.loc["2024-12", "tirex2_raw"]),
            "dec_sa": ru(bm.loc["2024-12", "tirex2_sa"]), "pro_apr": ru(bm.loc["2024-04", "prophet_default"]),
            "ens_apr": ru(bm.loc["2024-04", "ens_main"]), "ens_lose": lose_text,
            "nodec_gap": ru(ch["разрыв с Prophet без декабря, %"]), "ens_nofm": ru(ch["MAE ансамбля без FM (панель, LightGBM, сглаживание)"]),
            "grid_lo": ru(ch["MAE Prophet: лучшая из 8 настроек сетки, 400 МО"]), "grid_hi": ru(ch["MAE Prophet: худшая из 8 настроек сетки, 400 МО"]),
            "r2_within": ru(ch["R² г/г внутри месяца цели, среднее по месяцам"], 2), "r2oos_growth": ru(ch["R²_oos к наивному с приростом г/г"], 2)}


def example_alert(name="Белозерский"):
    """Пример одной тревоги для презентации: МО из разбора паводка, числа — из outputs/case_flood2024.csv."""
    d = pd.read_csv(OUTPUTS / "case_flood2024.csv")
    r = d[d["МО"].str.contains(name)].iloc[0]
    return {"ex_msgs": int(r["сообщений"]), "ex_first": day_month(r["первое сообщение"]),
            "ex_mp_apr": ru(r["Маркетплейсы апрель, %"], 0, True), "ex_mp_may": ru(r["Маркетплейсы май, %"], 0, True)}


def nums():
    """Числа результатов для текста страницы и презентации — из итоговых таблиц outputs/. Размеры корпуса текстов и
    итоги ручной проверки выборок заданы в разметке."""
    m = pd.read_csv(OUTPUTS / "tables" / "forecast_metrics.csv")
    a = m[(m["МО"] == "все МО") & (m["окно"] == "все точки") & (m["категория"] == "Все категории")].set_index("модель")
    sub = m[(m["МО"] == "400 МО") & (m["окно"] == "все точки")].pivot_table(index="модель", columns="категория", values="MAE")
    dev = m[(m["МО"] == "все МО") & (m["окно"] == "разработка 01–06") & (m["категория"] == "Все категории")].set_index("модель").MAE
    ctl = m[(m["МО"] == "все МО") & (m["окно"] == "контроль 07–11") & (m["категория"] == "Все категории")].set_index("модель")
    ctl_dm = [float(x) for x in ctl.loc["ens_main", "DM p (h=1/2/3)"].split("/")]
    cats = m[(m["МО"] == "все МО") & (m["окно"] == "все точки")].pivot_table(index="модель", columns="категория", values="MAE")
    ens, pro = a.loc["ens_main", "MAE"], a.loc["prophet_default", "MAE"]
    fmm = mae_of(["tirex2_raw", "tirex2", "tirex2_sa", "ses_sa", "timesfm25", "timesfm25_sa", "chronos2_sa", "chronos2_sa_cross",
                  "chronos2_sa_cov_cross"])
    cov, ref = fmm["chronos2_sa_cov_cross"], fmm["chronos2_sa_cross"]
    cov_text = (f"почти не помогают: {ru(cov)} против {ru(ref)} без них." if abs(cov / ref - 1) < 0.01 else
                f"{'помогают' if cov < ref else 'мешают'}: {ru(cov)} против {ru(ref)} без них.")
    r = pd.read_csv(OUTPUTS / "detectors_synthetic.csv").groupby("детектор")[["VUS-PR", "полнота@3%", "NAB-подобная", "F1 с допуском"]].mean()
    ds = pd.read_csv(OUTPUTS / "detectors_synthetic.csv")
    rs = ds[ds["форма"] == "отраслевой"].groupby("детектор")["VUS-PR"].mean()   # шок только в маркетплейсах и транспорте
    ac = pd.read_csv(OUTPUTS / "tables" / "alarm_clusters.csv").set_index("связь")   # кластеры тревог 2024 г. (материал 04)
    orsk = pd.read_csv(OUTPUTS / "case_flood2024.csv").set_index("МО").loc["городской округ город Орск"]   # карточка-пример (материал 08)
    dm = [ru(float(x), 3 if float(x) < 0.1 else 2) for x in a.loc["ens_main", "DM p (h=1/2/3)"].split("/")]
    dose = pd.read_csv(OUTPUTS / "dose_response_summary.csv").set_index("группа")
    w = pd.read_csv(OUTPUTS / "tables" / "warning_llm.csv").set_index("вариант")
    auc = pd.read_csv(OUTPUTS / "tables" / "warning_auc_llm.csv").set_index("модель подавления")
    src = pd.read_csv(OUTPUTS / "case_flood2024_sources.csv").set_index("источники")
    tl = pd.read_csv(OUTPUTS / "case_flood2024_timeline.csv")
    iv = pd.read_csv(OUTPUTS / "tables" / "forecast_intervals.csv")
    iv = iv[(iv["модель"] == "ens_main") & (iv["категория"] == "Все категории")].set_index("h")
    db = detect_block()
    wk = db["weekly"]
    fd = pd.read_csv(OUTPUTS / "tables" / "detectors_flood.csv")
    fg = pd.read_csv(OUTPUTS / "tables" / "detectors_flood_group.csv").set_index("категория")
    fs = pd.read_csv(OUTPUTS / "tables" / "detectors_flood_group_sources.csv").set_index("группа")
    n_fs = int(fs.filter(regex=": p$").notna().sum().sum())                        # все проверки таблицы устойчивости
    end = pd.Timestamp("2024-04-30")
    lead_sys = (end - tl[["первое сообщение: сайт МЧС", "акт опубликован"]].apply(pd.to_datetime).min(axis=1)).dt.days
    lead_tg = (end - tl.filter(like="первое сообщение").apply(pd.to_datetime).min(axis=1)).dt.days
    cf = pd.read_csv(OUTPUTS / "case_flood2024.csv")
    site_mo = cf[cf["первое сообщение на сайте МЧС"].notna()]                     # районы, названные на сайтах МЧС
    site_lead = (end - pd.to_datetime(site_mo["первое сообщение на сайте МЧС"])).dt.days
    n5 = int(fg.loc["Маркетплейсы", "МО"])
    Y, ids, months, meta = pnl.load()
    n_fc = len(ids) * sum(min(3, 12 - t) for t in range(1, 12))                  # точки янв–ноя, цели до декабря
    tg = [100 * (1 - sub.loc["prophet_tuned", c] / sub.loc["prophet_default", c]) for c in pnl.CATS[1:]]   # подбор по категориям
    bk = pd.read_csv(OUTPUTS / "offline_breaks.csv")
    bmp = bk[(bk["месяц разладки"] == "2024-02") & (bk["категория"] == "Маркетплейсы")]
    apr23 = bk[(bk["месяц разладки"] == "2023-04") & (bk["категория"] == "Все категории")].territory_id.nunique()
    ctx = [c["values"] for c in fm_block()["context"]["series"] if c["name"].endswith("*")]   # FM на входе без сезонности
    ft = a.loc[["chronos2_sa_ft", "chronos2_sa_ft_lr1e5"], "MAE"]                # дообучение всех весов Chronos-2
    fl_hit, fl_site_hit = round(fd["5+ сообщений: апрель"].max() * n5), round(fd["на сайте МЧС: апрель"].max() * len(site_mo))
    tg_reg = cf.groupby("регион")["Telegram: новости"].transform(lambda x: (x > 0).any())   # регионы с новостными лентами
    acts = pd.read_parquet(PROCESSED / "events.parquet").drop_duplicates("eo")
    act_base = w.loc["только с актом", "Маркетплейсы: точность"] / w.loc["только с актом", "Маркетплейсы: подъём"]   # база тех же МО
    ev = pd.read_csv(OUTPUTS / "tables" / "events2024.csv")
    ev_q = lambda sel, ctrl: ev[(ev["источник"] == "официальные каналы") & (ev["события"] == sel) & (ev["контроль"] == ctrl)].iloc[0]
    ev_main = ev_q("тяжёлые (масштаб ≥ 100), основной тест", "тот же регион")
    ev_null = ev_q("тяжёлые (масштаб ≥ 100), основной тест", "регионы без событий")
    ev_all = ev_q("все события", "регионы без событий")
    ev_dose = pd.read_csv(OUTPUTS / "tables" / "events2024_dose.csv").set_index("мера")
    ev_channels = len(pd.read_csv(ROOT / "configs" / "official_channels.csv"))
    n = {"ens": ru(ens), "prophet": ru(pro), "vs_prophet": ru(100 * (1 - ens / pro)), "better_share": ru(100 * a.loc["ens_main", "доля МО лучше prophet_default"]),
         "r2_ens": ru(a.loc["ens_main", "R² г/г"], 2), "r2_prophet": ru(a.loc["prophet_default", "R² г/г"], 2, True),
         "cov1": ru(100 * iv.loc[1, "покрытие 80%"]), "cov3": ru(100 * iv.loc[3, "покрытие 80%"]),
         "wid1": ru(iv.loc[1, "ширина 80%, %"]), "wid3": ru(iv.loc[3, "ширина 80%, %"]), "pro_dev": ru(dev["prophet_default"]), "pro_ctl": ru(ctl.loc["prophet_default", "MAE"]),
         "ens_dev": ru(dev["ens_main"]), "ens_ctl": ru(ctl.loc["ens_main", "MAE"]),
         "panel": ru(a.loc["panel", "MAE"]), "timecast": ru(sub.loc["prophet_timecast", "Все категории"]),
         "nn_tcn": ru(a.loc["tcn_sa", "MAE"]), "nn_patch": ru(a.loc["patchtst_sa", "MAE"]), "nn_lstm": ru(a.loc["lstm_sa", "MAE"]),
         "tuned_mp": ru(sub.loc["prophet_tuned", "Маркетплейсы"]), "ens_mp_400": ru(sub.loc["ens_main", "Маркетплейсы"]),
         "tirex_raw": ru(fmm["tirex2_raw"]), "tirex_dev": ru(fmm["tirex2"]), "tirex_sa": ru(fmm["tirex2_sa"]), "tfm_sa": ru(fmm["timesfm25_sa"]),
         "ft_lo": ru(100 * (ft.min() / a.loc["chronos2_sa", "MAE"] - 1)), "ft_hi": ru(100 * (ft.max() / a.loc["chronos2_sa", "MAE"] - 1)),
         "ft_dev_x": ru(dev[ft.index].min() / dev["chronos2_sa"], 1),
         "tfm_raw": ru(fmm["timesfm25"]), "ses_sa": ru(fmm["ses_sa"]), "ses_gap": ru(100 * (fmm["ses_sa"] / fmm["timesfm25_sa"] - 1), 1), "chr_sa": ru(fmm["chronos2_sa"]), "chr_cross": ru(fmm["chronos2_sa_cross"]),
         "glr_vus": ru(r.loc["6 категорий: GLR", "VUS-PR"], 3), "ensr_vus": ru(r.loc["ансамбль рангов", "VUS-PR"], 3),
         "sel_rec": ru(100 * r.loc[CHOSEN, "полнота@3%"]), "sel_nab": ru(r.loc[CHOSEN, "NAB-подобная"], 1),
         "ens_nab_lo": ru(r.filter(like="ансамбль", axis=0)["NAB-подобная"].min(), 1), "ens_nab_hi": ru(r.filter(like="ансамбль", axis=0)["NAB-подобная"].max(), 1),
         "glr1_vus": ru(r.loc["GLR (окно 3)", "VUS-PR"], 3),
         "sec_glr6": ru(rs.loc["6 категорий: GLR"], 3), "sec_glr1": ru(rs.loc["GLR (окно 3)"], 3),
         "glr_f1": ru(r.loc["6 категорий: GLR", "F1 с допуском"], 3), "sel_f1": ru(r.loc[CHOSEN, "F1 с допуском"], 3),
         "cl_self": ru(ac.loc["повторение в том же МО через месяц", "отношение частот"], 2),
         "cl_self_perm": ru(ac.loc["повторение в том же МО через месяц", "при перестановках"], 2),
         "cl_self_p": ru(ac.loc["повторение в том же МО через месяц", "p"], 3),
         "cl_nb": ru(ac.loc["сосед с тревогой в том же месяце", "отношение частот"], 2),
         "cl_nb_perm": ru(ac.loc["сосед с тревогой в том же месяце", "при перестановках"], 2),
         "cl_nb_p": ru(ac.loc["сосед с тревогой в том же месяце", "p"], 3),
         "cl_region": ru(ac.loc["другой МО региона с тревогой в том же месяце", "отношение частот"], 2),
         "cl_region_p": ru(ac.loc["другой МО региона с тревогой в том же месяце", "p"], 2),
         "cl_prev": ru(ac.loc["сосед с тревогой в прошлом месяце", "отношение частот"], 2),
         "cl_prev_p": ru(ac.loc["сосед с тревогой в прошлом месяце", "p"], 2),
         "orsk_msgs": plural(int(orsk["сообщений"]), "сообщение", "сообщения", "сообщений"), "orsk_first": day_month(orsk["первое сообщение"]),
         "orsk_mp": ru(orsk["Маркетплейсы апрель, %"], 1),
         "tg_lo": ru(min(tg)), "tg_hi": ru(max(tg)), "mp_gap_tuned": ru(100 * (sub.loc["ens_main", "Маркетплейсы"] / sub.loc["prophet_tuned", "Маркетплейсы"] - 1)),
         "hl_ens": ru(sub.loc["ens_main", "Здоровье"]), "hl_tuned": ru(sub.loc["prophet_tuned", "Здоровье"]),
         "tuned_all": ru(sub.loc["prophet_tuned", "Все категории"]), "prophet_400": ru(sub.loc["prophet_default", "Все категории"]),
         "snaive_g": ru(a.loc["snaive_growth", "MAE"]), "vs_snaive": ru(100 * (1 - ens / a.loc["snaive_growth", "MAE"])),
         "dm1": dm[0], "dm2": dm[1], "dm3": dm[2], "n_fc": ru(n_fc), "n_fc_all": ru(6 * n_fc),
         "all_p": ru(dose.loc["акты с установленной причиной", "Все: p"], 2),
         "mp_list_p": ru(dose.loc["охват: перечень", "Маркетплейсы: p"], 3), "n_list": int(dose.loc["охват: перечень", "актов"]),
         "prec_act": ru(100 * w.loc["только с актом", "Маркетплейсы: точность"], 1), "n_ep_act": ru(w.loc["только с актом", "эпизодов"]),
         "hits_act": ru(w.loc["только с актом", "Маркетплейсы: точность"] * w.loc["только с актом", "эпизодов"]),
         "chance_act": ru(w.loc["только с актом", "эпизодов"] * act_base),
         "mp": ru(dose.loc["акты с установленной причиной", "Маркетплейсы: медиана"], 1), "mp_p": ru(dose.loc["акты с установленной причиной", "Маркетплейсы: p"], 3),
         "mp_list": ru(dose.loc["охват: перечень", "Маркетплейсы: медиана"], 1),
         "lift_act": ru(w.loc["только с актом", "Маркетплейсы: подъём"], 2), "lift_plac": ru(w.loc["плацебо: только с актом", "Маркетплейсы: подъём"], 2),
         "lift_news": ru(w.loc["только с новостями МЧС", "Маркетплейсы: подъём"], 2),
         "base_rate": ru(100 * w.loc["базовая частота (все МО)", "Маркетплейсы: точность"], 1), "n_ep": ru(w.loc["все эпизоды", "эпизодов"]),
         "base_act": ru(100 * act_base, 1),
         "wk_ep": plural(episodes([x for x in wk if known(x) and "эхо" not in x["why"]]), "эпизод", "эпизода", "эпизодов"),
         "wk_all": len(wk), "wk_echo": sum("эхо" in x["why"] for x in wk),
         "wk_unk": sum(not known(x) for x in wk), "wk_ok": sum(known(x) and "эхо" not in x["why"] for x in wk),
         "ens_gap": ru(max(100 * (cats.loc["ens_main", c] / cats[c].min() - 1) for c in cats.columns), 1),
         "fl_n5": n5, "fl_hit": fl_hit,
         "fl_mp_p": ru(fg.loc["Маркетплейсы", "p (Манн–Уитни)"], 3),
         "fl_all_p": ru(fg.loc["Все категории", "p (Манн–Уитни)"], 2), "fl_mp_z": ru(fg.loc["Маркетплейсы", "медиана −z"], 2),
         "rep_pages_text": plural(len(PdfReader(ROOT / "report" / "methodology.pdf").pages), "страница", "страницы", "страниц"),
         "cov_text": cov_text,
         "mp_break": len(bmp), "mp_break_down": int((bmp["сдвиг, %"] < 0).sum()),
         "br_apr23": ru(apr23), "br_apr23_share": ru(100 * apr23 / len(ids)),

         **example_alert(),
         "flood_rho": ru(src.loc["сайты МЧС", "Все: ρ"], 2), "flood_p": ru(src.loc["сайты МЧС", "Все: p"], 3),
         "flood_lead": int(lead_sys.max()), "flood_lead_tg_min": int(lead_tg.min()), "flood_lead_tg": int(lead_tg.max()), "n_mo": ru(len(ids)),
         "lag_lo": ru(db["lag_lo"], 2), "lag_hi": ru(db["lag_hi"], 2),
         **prior_nums(ds),
         "sel_vus": ru(r.loc[CHOSEN, "VUS-PR"], 3), "ens_wape": ru(a.loc["ens_main", "WAPE, %"], 1),
         "fm_win": min(sum(v < 1 for v in c) for c in ctx), "fm_pts": len(ctx[0]),
         "dose_cells": int(src.filter(like=": ρ").size), "dose_neg": int((src.filter(like=": ρ") < 0).sum().sum()),
         "rec5": ru(100 * ds[(ds["размер"] == 0.05) & (ds["детектор"] == CHOSEN)]["полнота@3%"].mean()),
         "n_events": int(((acts.action != "иное") & ~acts.old).sum()),   # новые события: без правок старых актов
         "n_acts_text": plural(int(dose.loc["акты с установленной причиной", "актов"]), "акт", "акта", "актов"),
         "n_exp": ru(0.05 * sum(dose[[c for c in dose.columns if c.endswith(": p")]].notna().sum()), 1),
         "flood_lead_min": int(lead_sys.min()),
         "fl_site_n": int(fs.loc["1+ сообщение на сайте МЧС", "МО"]), "fl_site_p": ru(fs.loc["1+ сообщение на сайте МЧС", "Маркетплейсы: p"], 3),
         "fl_site_bonf": ru(fs.loc["1+ сообщение на сайте МЧС", "Маркетплейсы: p с поправкой"], 3), "fl_src_tests": n_fs,
         "fl_mchs_n": int(fs.loc["1+ сообщение МЧС", "МО"]), "fl_mchs_p": ru(fs.loc["1+ сообщение МЧС", "Маркетплейсы: p"], 4),
         "fl_mchs_bonf": ru(fs.loc["1+ сообщение МЧС", "Маркетплейсы: p с поправкой"], 3),
         "auc_cv": ru(auc.loc["перекрёстная проверка по месяцам", "AUC"], 2), "auc_online": ru(auc.loc["онлайн", "AUC"], 2),
         "fl_health_p": ru(fg.loc["Здоровье", "p (Манн–Уитни)"], 3),
         "fl_site_hit": fl_site_hit,
         "lift_addr": ru(w.loc["адресные (акт по перечню или новость с МО)", "Маркетплейсы: подъём"], 2),
         "dm_ctl": f"{ru(min(ctl_dm), 2)}–{ru(max(ctl_dm), 2)}",
         "fl_site_mp_med": ru(site_mo["Маркетплейсы апрель, %"].median(), 1, True), "fl_site_mp_abs": ru(-site_mo["Маркетплейсы апрель, %"].median(), 1),
         "fl_none_mp_med": ru(cf.loc[cf["сообщений"] == 0, "Маркетплейсы апрель, %"].median(), 1, True), "fl_site_lead_med": int(site_lead.median()),
         "fl_news_mo": int((cf["Telegram: новости"] > 0).sum()), "fl_news_all": int(tg_reg.sum()),
         "fl_first": day_month(tl["первое сообщение: сайт МЧС"].min()),
         "lift_lo": ru(w.loc["только с актом", "Маркетплейсы: подъём, 2,5%"], 2), "lift_hi": ru(w.loc["только с актом", "Маркетплейсы: подъём, 97,5%"], 2),
         "n_clu": int(w.loc["только с актом", "кластеров регион × месяц"]),
         "mp_all_bonf": ru(dose.loc["акты с установленной причиной", "Маркетплейсы: p с поправкой"], 2),
         "mp_list_bonf": ru(dose.loc["охват: перечень", "Маркетплейсы: p с поправкой"], 2),
         "mp_list_bonf_all": ru(dose.loc["охват: перечень", "Маркетплейсы: p с поправкой на все проверки"], 2),
         "mp_flood": ru(dose.loc["паводок/наводнение", "Маркетплейсы: медиана"], 1), "mp_flood_p": ru(dose.loc["паводок/наводнение", "Маркетплейсы: p"], 3),
         "n_tests": int(sum(dose[[c for c in dose.columns if c.endswith(": p")]].notna().sum())),
         "n_sig": int(sum((dose[[c for c in dose.columns if c.endswith(": p")]] < 0.05).sum())),
         "ev_channels": ru(ev_channels), "ev_total": ru(int(ev_all["событий"])), "ev_heavy": ru(int(ev_null["событий"])),
         "ev_test_n": ru(int(ev_main["событий"])), "ev_mo": ru(int(ev_main["МО"])),
         "ev_z": ru(ev_main["z Стауффера"], 2, True), "ev_p": ru(ev_main.p, 2), "ev_p_bonf": ru(ev_main["p с поправкой"], 2),
         "ev_z_null": ru(ev_null["z Стауффера"], 2), "ev_rho_scale": ru(ev_dose.loc["масштаб", "ρ"], 3),
         "ev_rho_posts": ru(ev_dose.loc["постов", "ρ"], 3), "ev_dose_mo": ru(int(ev_dose.loc["масштаб", "МО-событий"])),
         **extra_forecast(ens)}
    return n


REPO_URL = "https://github.com/vakuznetzovaclaud/sberindex"


def main():
    data = {"forecast": forecast_block(), "fm": fm_block(), "detect": detect_block(), "text": text_block(), "feed": feed_block(),
            "case": case_block(), "nums": nums()}
    html = (SITE / "mag.html").read_text(encoding="utf-8")
    for k, v in data["nums"].items():
        html = html.replace("{{" + k + "}}", str(v))
    html = (html.replace("{{repo_url}}", REPO_URL) if REPO_URL else
            re.sub(r'\s*<a\b[^>]*href="\{\{repo_url\}\}"[^>]*>.*?</a>', "", html, flags=re.S))   # без адреса — без ссылки
    html = html.replace("{{ARCH}}", (ROOT / "report" / "fig" / "architecture.svg").read_text(encoding="utf-8"))
    html = html.replace("{{FONT}}", base64.b64encode((SITE / "NunitoSans.ttf").read_bytes()).decode())
    html = html.replace("/*CSS*/", (SITE / "mag.css").read_text(encoding="utf-8"))
    left = re.findall(r"{{\w+}}", html)
    assert not left, f"не заполнены: {left}"
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float).replace("</", "<\\/")
    app = (SITE / "charts.js").read_text(encoding="utf-8") + (SITE / "mag.js").read_text(encoding="utf-8")
    out = html.replace("/*DATA*/null", blob).replace("/*APP*/", app)
    (SITE / "index.html").write_text(out, encoding="utf-8")
    deck = (SITE / "deck.html").read_text(encoding="utf-8")        # презентация: те же числа, данные и графики
    for k, v in data["nums"].items():
        deck = deck.replace("{{" + k + "}}", str(v))
    deck = deck.replace("{{ARCH}}", (ROOT / "report" / "fig" / "architecture.svg").read_text(encoding="utf-8"))
    deck = (deck.replace("{{repo_url_text}}", REPO_URL) if REPO_URL else
            re.sub(r'[ \t]*<p\b[^>]*>(?:(?!</?p\b).)*\{\{repo_url_text\}\}(?:(?!</?p\b).)*</p>\n?', "", deck, flags=re.S))
    deck = deck.replace("{{FONT}}", base64.b64encode((SITE / "NunitoSans.ttf").read_bytes()).decode())
    left = re.findall(r"{{\w+}}", deck)
    assert not left, f"презентация: не заполнены {left}"
    deck = deck.replace("/*DATA*/null", blob).replace("/*APP*/", (SITE / "charts.js").read_text(encoding="utf-8") + (SITE / "deck.js").read_text(encoding="utf-8"))
    (SITE / "presentation.html").write_text(deck, encoding="utf-8")
    # копия отчёта рядом со страницей: кнопка «Методологический отчёт, PDF» в index.html ссылается на неё относительно
    shutil.copy(ROOT / "report" / "methodology.pdf", SITE / "methodology.pdf")
    src = "".join((SITE / f).read_text(encoding="utf-8") for f in ("mag.html", "deck.html", "mag.js", "deck.js", "charts.js"))
    unused = sorted(set(data["nums"]) - set(re.findall(r"{{(\w+)}}", src)) - set(re.findall(r"\bN\.(\w+)", src)))
    if unused:
        print("числа, которых нет в разметке:", ", ".join(unused))
    print(f"site/index.html: {len(out) / 1e6:.2f} МБ")


if __name__ == "__main__":
    main()
