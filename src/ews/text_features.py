"""Текстовые признаки МО-месяца с датой знания: режимы ЧС и повышенной готовности из официальных актов и новости
главных управлений МЧС о последствиях. Признак месяца t собран только из текстов, опубликованных до конца месяца t,
поэтому годится и как ковариата прогноза с точки T0 = t, и как априор детектора в месяце t."""
import numpy as np
import pandas as pd

from . import news_classify as nc, panel as pnl
from .geo import Matcher
from .news_mchs import DIR as MCHS_DIR
from .paths import INTERIM, PROCESSED

# причины (ключи acts.CAUSES), которые по смыслу могут сдвинуть потребление; засухи, эпизоотии и землетрясения без
# разрушений — вне списка. Порядок списка — порядок признаков модели подавления (scripts/eval_warning.py).
RELEVANT = ["паводок/наводнение", "лесные/природные пожары", "авария ЖКХ/тепло-, энерго-, водоснабжение",
            "техногенная авария/разлив/обрушение/взрыв", "метеоявления (ветер, снег, мороз)", "атаки БПЛА/обстрелы"]
# Для признаков — не слова об угрозе (МЧС каждый день предупреждает о ветре и пишет о бытовых пожарах), а слова о
# последствиях и о введении режима: подтопление, эвакуация, отключения, разрушения, пострадавшие, режим ЧС.
ANY_IMPACT = (r"подтоп|затоп|вода зашла|вышла из берег|эвакуир|отселен|обесточ|без света|без тепла|без электр|без воды|"
              r"разруш|поврежд|ущерб|пострадав|обрушен|чрезвычайн\w* ситуац|режим\w* (?:чс|чрезвычайн)|повышенн\w* готовност")
NOT_EVENT = nc.NOT_EVENT + r"|пострадавших нет|без пострадавших|не поступ|не зарегистр|не зафикс|нет подтоп"
# бытовые пожары (с эвакуацией жильцов, повреждённой техникой) — главная тема новостей МЧС, на потребление МО не влияют
HOUSE_FIRE, WILDFIRE = r"пожар|огн[её]м|огонь|возгоран|задымл|сгорел|горел", r"лесн|природн|ландшафтн|(?<![а-яё])палы|пал трав"
FEATURES = ["act_new", "act_local", "act_recent", "act_people", "news_mo", "news_region", "news_region_anom"]


def news_links(rebuild=False, source="llm"):
    """Новости МЧС о реальных событиях и МО, названные в них: (region_code, id, dt, territory_id, title);
    territory_id = -1 — событие без названного МО панели (учитывается в региональном счётчике).
    source="llm" — заголовки, которые модель отметила как массовое событие (news_classify), МО ищутся в заголовке,
    в выписанных моделью местах и в тексте новости; source="regex" — словарь последствий без модели (для абляции)."""
    f = INTERIM / f"mchs_event_links_{source}.parquet"
    inputs = list((MCHS_DIR / "titles").glob("titles_*.parquet")) + ([nc.path()] if source == "llm" and nc.path().exists() else [])
    inputs += list((MCHS_DIR / "bodies").glob("*.txt")) if source == "llm" else []
    fresh = f.exists() and all(p.stat().st_mtime <= f.stat().st_mtime for p in inputs)   # кэш новее всех входов
    if fresh and not rebuild:
        return pd.read_parquet(f)
    M = Matcher()
    rows = []
    if source == "llm":
        lab = nc.labels()
        lab = lab[lab.apply(nc.is_event, axis=1)]
        for r in lab.itertuples():
            b = nc.body_path(r.region_code, r.id)
            text = r.title + " ; " + " ; ".join(r.места or []) + (" ; " + b.read_text(encoding="utf-8") if b.exists() else "")
            tids = M.match(text, r.region_code) or {-1}
            rows += [(int(r.region_code), int(r.id), r.dt, int(tid), r.title) for tid in tids]
    else:
        for p in sorted((MCHS_DIR / "titles").glob("titles_*.parquet")):
            t = pd.read_parquet(p)
            has = lambda rx: t.title.str.contains(rx, case=False, regex=True)
            t = t[has(ANY_IMPACT) & ~has(NOT_EVENT) & ~(has(HOUSE_FIRE) & ~has(WILDFIRE))]
            for r in t.itertuples():
                tids = M.match(r.title, r.region_code) or {-1}
                rows += [(int(r.region_code), int(r.id), r.dt, int(tid), r.title) for tid in tids]
    d = pd.DataFrame(rows, columns=["region_code", "id", "dt", "territory_id", "title"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    d.to_parquet(f)
    return d


def covered_regions():
    """Регионы, по которым собран архив новостей МЧС (файл есть и не пуст)."""
    out = set()
    for p in (MCHS_DIR / "titles").glob("titles_*.parquet"):
        if len(pd.read_parquet(p, columns=["id"])):
            out.add(int(p.stem.split("_")[1]))
    return out


NOWCAST = ["act_new", "act_local", "act_people", "news_mo", "news_region"]


def monthly(rebuild=False, source="llm", day=None):
    """{признак: массив [МО × месяц]} в порядке МО и месяцев панели. Новостные признаки — NaN в регионах без архива МЧС.
    day — только тексты, опубликованные с 1-го по day-е число месяца (для наукаста: прогноз делается в середине
    месяца, когда муниципальных трат за него ещё нет, а акты и новости уже вышли)."""
    _, ids, months, meta = pnl.load()
    pos = {t: k for k, t in enumerate(ids)}
    mi = {m: k for k, m in enumerate(months)}
    N, T = len(ids), len(months)
    out = {f: np.zeros((N, T)) for f in FEATURES}

    ev = pd.read_parquet(PROCESSED / "events.parquet")
    # акты «иное» (выплаты, резервный фонд) и изменения давних актов — не новые события
    ev = ev[ev.cause.isin(RELEVANT) & ev.action.isin(["введение", "изменение"]) & ~ev.old & ev.territory_id.notna()]
    if day:
        ev = ev[pd.to_datetime(ev.pub_date).dt.day <= day]
    for r in ev.itertuples():
        i, m = pos.get(int(r.territory_id)), mi.get(str(r.pub_date)[:7])
        if i is None or m is None:
            continue
        out["act_new"][i, m] += 1
        out["act_local"][i, m] += r.scope == "перечень"
        out["act_people"][i, m] = max(out["act_people"][i, m], np.log1p(float(r.people or 0)))
    out["act_recent"] = out["act_new"] + np.pad(out["act_new"], ((0, 0), (1, 0)))[:, :T]

    links = news_links(rebuild, source)
    if day:
        links = links[links.dt.str[8:10].astype(int) <= day]
    links = links.assign(m=links.dt.str[:7].map(mi)).dropna(subset=["m"])
    links["m"] = links.m.astype(int)
    per_mo = links[links.territory_id >= 0].groupby(["territory_id", "m"]).id.nunique()
    for (tid, m), n in per_mo.items():
        if tid in pos:
            out["news_mo"][pos[tid], m] = n
    per_reg = links.drop_duplicates(["region_code", "id"]).groupby(["region_code", "m"]).size().unstack(fill_value=0)
    per_reg = per_reg.reindex(columns=range(T), fill_value=0)
    reg = meta.region_code.astype(int).to_numpy()
    cov = covered_regions()
    for k, rc in enumerate(reg):
        if rc not in cov:
            out["news_mo"][k] = np.nan
            out["news_region"][k] = np.nan
            out["news_region_anom"][k] = np.nan
            continue
        cnt = per_reg.loc[rc].to_numpy(float) if rc in per_reg.index else np.zeros(T)
        out["news_region"][k] = np.log1p(cnt)
        prev = np.array([cnt[max(0, m - 12):m].mean() if m > 0 else np.nan for m in range(T)])
        out["news_region_anom"][k] = np.log1p(cnt) - np.log1p(prev)       # всплеск против среднего прошлых месяцев
    return {k: out[k] for k in NOWCAST} if day else out
