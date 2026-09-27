"""Реестр событий-шоков: акты о режимах ЧС и повышенной готовности превращаются в строки «событие × МО» с датой знания,
типом причины, уровнем и территорией. Территория: акт на весь субъект относится ко всем МО региона в панели; иначе — МО и населённые
пункты, названные в акте (сопоставление по справочнику и ОКТМО).
Причина — правила по дословной причине, которую выписала модель; если там пусто — по названию акта; у акта о внесении
изменений без причины она наследуется от исходного акта, если тот есть в реестре (ссылка «от ДД.ММ.ГГГГ № N» или
«от 3 июля 2024 года № N»). Акты о выплатах, резервном фонде, субсидиях и правилах поведения событиями не считаются."""
import json
import re

import pandas as pd

from . import acts
from .geo import Matcher
from .paths import PROCESSED


def _causes(ex, lst):
    """Причина каждого акта: дословная причина, затем название, затем исходный акт (для изменений)."""
    key = {}
    for eo, rc, title in zip(lst.index, lst.region_code, lst.title):
        own = _refs(title)[:1]                          # первая ссылка в названии — реквизиты самого акта
        if own:
            key[(rc, *own[0])] = eo
    cause = {}
    for r in ex.itertuples():
        c = acts.cause_category(getattr(r, "причина_дословно", "") or "")
        if c == acts.OTHER:
            c = acts.cause_category(lst.loc[r.eo, "title"])
        cause[r.eo] = c
    for r in ex.itertuples():
        a = lst.loc[r.eo]
        if cause[r.eo] == acts.OTHER and a.action == "изменение":
            for d, n in _refs(a.title)[1:]:
                src = key.get((a.region_code, d, n))
                if src in cause and cause[src] != acts.OTHER:
                    cause[r.eo] = cause[src]
                    break
    return cause


MONTHS = {m: k + 1 for k, m in enumerate(["январ", "феврал", "март", "апрел", "ма", "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр"])}
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
WORD_DATE = re.compile(rf"(\d{{1,2}})\s+({MONTH_RE})\w*\s+(\d{{4}})")          # «06 июля 2021», «12 августа 2022»
MONTH_YEAR = re.compile(rf"\b({MONTH_RE})\w*\s+(\d{{4}})\s*г")                  # «в августе 2023 г(ода)»


def _dates(text):
    """Даты в тексте по порядку: «ДД.ММ.ГГГГ», «ДД месяца ГГГГ», «месяц ГГГГ года» (первое число месяца)."""
    out = [(m.start(), pd.to_datetime(m.group(1), format="%d.%m.%Y", errors="coerce")) for m in re.finditer(r"(\d{1,2}\.\d{1,2}\.\d{4})", text)]
    out += [(m.start(), pd.Timestamp(int(m.group(3)), MONTHS[m.group(2)], min(int(m.group(1)), 28))) for m in WORD_DATE.finditer(text)]
    taken = {p for p, _ in out}
    out += [(m.start(), pd.Timestamp(int(m.group(2)), MONTHS[m.group(1)], 1)) for m in MONTH_YEAR.finditer(text)
            if not any(abs(m.start() - p) < 4 for p in taken)]
    return [d for _, d in sorted(out) if pd.notna(d)]


REF = re.compile(rf"от\s+(\d{{1,2}}\.\s*\d{{1,2}}\.\s*\d{{4}}|\d{{1,2}}\s+(?:{MONTH_RE})\w*\s+\d{{4}})\s*(?:года|г\.?)?\s*(?:№|No)\s*([\w\-/]+)")


def _refs(title):
    """Реквизиты правовых актов в названии по порядку: (дата, номер); дата цифрами («от 03.07.2024 № 45-пг») или
    словами («от 3 июля 2024 года № 45-пг»)."""
    out = []
    for d, n in REF.findall(title):
        m = re.match(rf"(\d{{1,2}})\s+({MONTH_RE})\w*\s+(\d{{4}})", d)
        t = (pd.Timestamp(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1))) if m else
             pd.to_datetime(re.sub(r"\s", "", d), format="%d.%m.%Y", errors="coerce"))
        if pd.notna(t):
            out.append((t, n))
    return out


def amends_old(title, pub_date, days=365):
    """Акт о внесении изменений в акт, принятый больше года назад (границы зон ЧС 2020–2021 гг., режимы COVID-19,
    приём вынужденных переселенцев 2022 г., порядки выплат 2021 г.): не новое событие, дата акта не совпадает с датой
    события. Первая дата в названии — реквизиты самого акта, вторая — изменяемого (цифрами или словами)."""
    ds = _dates(title)[1:]
    return bool(ds and (pd.Timestamp(pub_date) - ds[0]).days > days)


def old_event(cause_text, pub_date, days=365):
    """Причина называет только события старше года (снос дома после обрушения 2022 г.): акт не о новом событии."""
    text = re.sub(r"от\s+(?:\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\s+\w+\s+\d{4})", " ", cause_text or "")   # реквизиты правовых актов — не даты событий
    ds = [d for d in _dates(text) if d.year > 1990]
    return bool(ds and all((pd.Timestamp(pub_date) - d).days > days for d in ds))


def build():
    ex = pd.DataFrame([json.loads(l) for l in open(acts.extracted_path())])
    ex = ex[ex["ошибка"].isna()] if "ошибка" in ex else ex
    lst = pd.read_parquet(acts.DIR / "acts_2023_2024.parquet").set_index("eo")
    meta = pd.read_parquet(PROCESSED / "mo_meta.parquet")
    by_region = meta.groupby(meta.region_code.astype(int)).groups
    cause = _causes(ex, lst)
    M = Matcher()
    rows = []
    for _, r in ex.iterrows():
        a = lst.loc[r.eo]
        action = "иное" if re.search(acts.NON_EVENT, a.title.lower()) else a.action
        rc = int(a.region_code)
        names = " ; ".join(list(r.get("муниципальные_образования") or []) + list(r.get("населённые_пункты") or []))
        tids = M.match(names, rc, strict=False) if names.strip() else set()
        scope = "перечень"
        if not tids and r.get("вся_территория_субъекта"):
            tids, scope = set(by_region.get(rc, [])), "весь субъект"
        elif not tids:
            scope = "не привязано"
        base = {"eo": r.eo, "region_code": rc, "pub_date": a.pub_date, "doc_date": a.doc_date, "action": action,
                "old": (action == "изменение" and amends_old(a.title, a.pub_date)) or old_event(r.get("причина_дословно"), a.pub_date),
                "regime": r.get("режим"), "level": r.get("уровень"), "cause": cause[r.eo],
                "cause_text": r.get("причина_дословно"), "scope": scope, "houses": r.get("число_пострадавших_домов") or 0,
                "people": r.get("число_пострадавших_людей") or 0}
        if tids:
            rows += [base | {"territory_id": t} for t in tids]
        else:
            rows.append(base | {"territory_id": None})
    ev = pd.DataFrame(rows)
    ev.to_parquet(PROCESSED / "events.parquet")
    return ev
