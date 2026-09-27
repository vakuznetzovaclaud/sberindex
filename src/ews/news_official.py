"""Официальные Telegram-каналы регионов — ГУ МЧС, губернатор или правительство, ЦУР — за 2024-02-01…2024-11-30:
сбор через веб-превью t.me/s (как в news_telegram), отбор постов о последствиях ЧС с названием МО и масштаб события
из текста (подтопленные дома и участки, эвакуированные, без света и тепла). Список каналов —
configs/official_channels.csv. Нужен для проверки, повторяется ли на других событиях 2024 г. сдвиг, найденный на
паводке: scripts/eval_events_2024.py. Тексты хранятся локально и в репозиторий не входят."""
import glob
import json
import re
from multiprocessing import Pool

import pandas as pd

from . import news_classify as nc, news_telegram as nt, text_features as tf
from .geo import Matcher
from .paths import RAW, ROOT

DIR = RAW / "telegram_official"
CHANNELS = ROOT / "configs" / "official_channels.csv"
SINCE, UNTIL = "2024-02-01", "2024-11-30"

# масштаб: числа рядом с домами и участками, эвакуированными, отключёнными от света и тепла
NUM = r"(\d[\d\s]{0,6}\d|\d)"
SCALE = {
    "дома": [NUM + r"\s*(?:жил(?:ых|ые|ой)\s+)?(?:дом(?:ов|а)?|домовладени\w*|приусадебн\w+\s+участк\w*|придомов\w+\s+территори\w*|двор\w*)",
             r"подтоплен\w*\s+(?:\w+\s+){0,3}?" + NUM + r"\s"],
    "эвакуированы": [r"эвакуирован\w*\s+(?:\w+\s+){0,3}?" + NUM + r"\s*(?:чел|жител)",
                     NUM + r"\s*(?:человек|жител\w*)\s+(?:\w+\s+){0,2}?эвакуирован"],
    "без света и тепла": [r"без\s+(?:электро|тепло|газо|водо)\w*\s+(?:\w+\s+){0,4}?" + NUM + r"\s*(?:чел|жител|абонент|дом)",
                          NUM + r"\s*(?:человек|жител\w*|абонент\w*)\s+(?:\w+\s+){0,3}?без\s+(?:электро|тепло|света)"],
}
# событие: последствия для жителей (как STRONG в разметке новостей) или сгоревшие дома и огонь у сёл
FIRE_NEAR = r"сгорел\w* (?:\d[\d\s]* )?(?:жил\w* )?(?:дом|строени)|огонь (?:подош|переки)\w*|угроз\w* (?:населенн|сел|пос)"
# никогда не событие: учения, плановые отключения, прогнозы затопления, пожары бань
NEVER = (r"учени|тренировк|по легенде|отработ|планов\w* (?:работ|отключ)|плановое отключ|возможн\w* (?:затоплен|подтоплен)|"
         r"зон\w* возможного|(?<![а-я])бан[ьяиюе]")
# отрицание без чисел («подтопленных домов нет», «не зафиксировано») — не событие
NEG = r"(?:домов|подтоплений|пострадавших|погибших|разрушений)\s+нет|нет подтопл|не зарегистр|не зафикс|не выявл|не допущ|не поступ"
ADVICE = r"профилакт|памятк|напомина|рекоменд|прогноз|ожида|возможн|предупрежда|будьте|соблюдайте"


def scale(text):
    """Максимум по шаблонам в каждой группе; годы и явно лишние числа отбрасываются."""
    t = (text or "").lower().replace("\xa0", " ")
    out = {}
    for k, pats in SCALE.items():
        vals = [int(re.sub(r"\s", "", m.group(1))) for p in pats for m in re.finditer(p, t)]
        vals = [v for v in vals if 0 < v < 200000 and not 1900 <= v <= 2030]
        out[k] = max(vals) if vals else 0
    return out


def collect(job):
    """Посты одного канала за [SINCE, UNTIL]: дозапись по страницам, продолжение с места остановки, отметка .done."""
    ch, rc, kind = job
    f, st, done = DIR / f"{ch}.jsonl", DIR / f"{ch}.state", DIR / f"{ch}.done"
    if done.exists():
        return ch
    before = int(st.read_text()) if st.exists() else nt._first_id_after(ch, "2024-12-01")
    n, empty = 0, 0
    with open(f, "a", encoding="utf-8") as fh:
        while before and before > 1:
            posts = nt._page(ch, before)
            if not posts:
                before -= 20
                empty += 1
                if empty > 50:                              # длинный пустой участок ленты
                    break
                continue
            empty = 0
            for p in posts:
                if SINCE <= p["dt"][:10] <= UNTIL:
                    fh.write(json.dumps(p | {"channel": ch, "region_code": rc, "kind": kind}, ensure_ascii=False) + "\n")
                    n += 1
            fh.flush()
            before = min(p["id"] for p in posts)
            st.write_text(str(before))
            if min(p["dt"] for p in posts)[:10] < SINCE:
                break
    done.write_text(str(n))
    return ch


def collect_all(workers=4):
    """Все каналы списка, по одному каналу на поток (пауза между запросами — в каждом потоке)."""
    DIR.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(CHANNELS)
    with Pool(workers) as pool:
        for ch in pool.imap_unordered(collect, list(c[["channel", "region_code", "kind"]].itertuples(index=False, name=None))):
            print("готово:", ch, flush=True)


def event_posts():
    """Посты о последствиях ЧС с названными МО региона канала: МО, масштаб поста (сумма групп)."""
    rows = []
    for f in glob.glob(str(DIR / "*.jsonl")):
        rows += [json.loads(line) for line in open(f, encoding="utf-8")]
    df = pd.DataFrame(rows).drop_duplicates(["channel", "id"])
    M, out = Matcher(), []
    for p in df.itertuples():
        t = (p.text or "").lower().replace("ё", "е")
        strong = bool(re.search(nc.STRONG + "|" + FIRE_NEAR, t))
        sc = scale(p.text)
        if not (strong or sum(sc.values()) > 0):
            continue
        if re.search(NEVER, t) or (re.search(NEG + "|" + nc.NEGATION, t) and sum(sc.values()) == 0):
            continue
        if re.search(nc.NOT_EVENT + "|" + ADVICE, t) and not strong:
            continue
        if re.search(tf.HOUSE_FIRE, t) and not re.search(tf.WILDFIRE + "|подтоп|затоп|обесточ|без света|без тепла|без электр", t):
            continue                                        # бытовой пожар
        mo = sorted(M.match(p.text, p.region_code))
        if mo:
            out.append({"channel": p.channel, "kind": p.kind, "region_code": int(p.region_code), "dt": p.dt, "month": p.dt[:7],
                        "mo": mo, **sc, "масштаб": sum(sc.values())})
    return pd.DataFrame(out)
