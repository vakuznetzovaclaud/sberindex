"""Привязка текста к муниципальным образованиям панели: названия МО и их центров из справочника СберИндекса и
населённые пункты из ОКТМО (Росстат, раздел 2: пункт относится к муниципальному району или округу по первым пяти цифрам кода).
Поиск — в пределах региона источника, с учётом падежных окончаний; реки, совпадающие с названиями городов, отсекаются."""
import re
from collections import defaultdict

import pandas as pd

from .paths import RAW, PROCESSED

RIVERS = {"Урал", "Ишим", "Тобол", "Иртыш", "Кама", "Волга", "Обь", "Лена", "Амур", "Бия", "Катунь", "Томь", "Уфа", "Белая",
          "Самара", "Ока", "Дон", "Кубань", "Терек", "Енисей", "Ангара", "Тавда", "Тура", "Исеть", "Миасс", "Сакмара"}
PREFIX = re.compile(r"^(г|п|пгт|рп|с|д|х|ст-ца|ст|аул|сл|мкр|кп|дп|нп|у|м|заимка|жд ст|ж/д ст)\s+", re.I)


ADJ = r"(?:ий|ый|ой|ого|ому|им|ым|ом|ая|яя|ую|юю|ое|ее|ие|ые|их|ых|ими|ыми|ей|ою)"
NOUN = r"(?:а|я|у|ю|ом|ем|ём|е|ы|и|ой|ей|ою|ью|ах|ях|ам|ям|ами|ями|ов|ев)?"
REGION_NEXT = re.compile(r"^\s+(?:кра|област|респ|автоном|федерал)", re.I)
DISTRICT_NEXT = re.compile(r"^\s+(?:район|р-н|округ|муниципальн|городск|сельск|улус|кожуун)", re.I)


def _is_adj(name):
    return bool(re.search(r"(?:ий|ый|ой|ое|ая|ее|ие)$", name)) and len(name) > 5


def _pattern(name):
    """Регулярное выражение для названия с падежными окончаниями: у прилагательных («Хабаровский», «Первомайское») —
    окончания прилагательного, у существительных («Орск», «Ставрополь») — окончания существительного. Так «Ставрополье»
    не находит Ставрополь, а «Хабаровского (края)» — город Хабаровск."""
    n = name.replace("ё", "е")
    if _is_adj(n):
        stem, end = n[:-2], ADJ
    else:
        stem = n[:-1] if n[-1] in "аяоеьйыи" and len(n) > 4 else n
        end = NOUN
    return r"(?<![А-Яа-яЁё])" + re.escape(stem) + end + r"(?![А-Яа-яЁё])"


def gazetteer():
    """{region_code: [(territory_id, название, это_река_тоже, это_пункт)]} для МО панели."""
    meta = pd.read_parquet(PROCESSED / "mo_meta.parquet")
    key = {str(o).replace("-", "")[:5]: tid for tid, o in meta["oktmo"].items() if isinstance(o, str)}
    out = defaultdict(list)
    city_names = defaultdict(set)
    for tid, r in meta.iterrows():
        city_names[int(r.region_code)].add(str(r.name_short))
    for tid, r in meta.iterrows():
        rc = int(r.region_code)
        center = PREFIX.sub("", str(r.center or "")).strip()
        # центр района, который сам — отдельный городской округ (Хабаровск для Хабаровского района), району не
        # приписываем: упоминание города — это город
        if center in city_names.get(rc, set()) and center != str(r.name_short):
            center = ""
        for n in {str(r.name_short), center}:
            if len(n) >= 4 and n != "None":
                out[rc].append((tid, n, n in RIVERS, False))
    f = RAW / "rosstat" / "oktmo.csv"
    if not f.exists():
        raise FileNotFoundError(f"нет {f}: справочник ОКТМО скачивает make data (ews.rosstat.oktmo)")
    cols = ["reg", "dist", "sett", "pt", "ctrl", "section", "name", "add", "desc", "chg", "chgtype", "adopted", "effective"]
    d = pd.read_csv(f, sep=";", header=None, names=cols, dtype=str)
    d = d[(d.section == "2") & (d.pt != "000") & d.name.str.match(PREFIX)]
    region_of = {tid: int(r.region_code) for tid, r in meta.iterrows()}
    for (reg, dist), g in d.groupby(["reg", "dist"]):
        tid = key.get(reg + dist)
        if tid is None:
            continue
        for n in set(PREFIX.sub("", x).strip() for x in g.name):
            if len(n) >= 5:
                out[region_of[tid]].append((tid, n, n in RIVERS, True))
    return out


class Matcher:
    def __init__(self):
        self.g = gazetteer()
        self.rx = {rc: [(tid, n, river, pt, re.compile(_pattern(n))) for tid, n, river, pt in items] for rc, items in self.g.items()}

    def match(self, text, region_code, strict=True):
        """Множество territory_id, упомянутых в тексте (в пределах региона). Упоминание пропускается, если за ним идёт
        «края/области/республики» (это регион, а не город) или если это река; название района-прилагательного
        засчитывается, только если за ним идёт «район/округ» (иначе «Хабаровский» путается с городом Хабаровском).
        strict=False — для перечней, которые модель уже выписала из акта по одному названию: там «Сергиевский» без
        слова «район» — это район."""
        t = (text or "").replace("ё", "е")
        hits = set()
        for tid, n, river, pt, rx in self.rx.get(int(region_code), []):
            for m in rx.finditer(t):
                after = t[m.end():m.end() + 20]
                if REGION_NEXT.match(after):
                    continue
                if river and re.search(r"(?:рек[аиеуой]|р\.|уровень|уровня)\s*$", t[max(0, m.start() - 12):m.start()].lower()):
                    continue
                adj = _is_adj(n.replace("ё", "е"))
                if strict and not pt and adj and not DISTRICT_NEXT.match(after):
                    continue
                if not adj and DISTRICT_NEXT.match(after):       # «в Хабаровском районе» — район, не город
                    continue
                hits.add(tid)
                break
        return hits
