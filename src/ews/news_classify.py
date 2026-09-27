"""Разметка заголовков новостей ГУ МЧС открытой моделью GigaChat3.1 (MIT, локально через Ollama): массовое событие
с последствиями для жителей или учения, прогноз, единичное происшествие; причина; места. Модель читает только
заголовок и ничего не дополняет по памяти. К модели идут заголовки, прошедшие словарный фильтр (слова об опасных явлениях
и последствиях), — так размечается 3–5% потока вместо всего архива."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import llm
from .news_mchs import DIR as MCHS_DIR, text
from .paths import INTERIM, config

HAZARD = (r"паводк|половод|наводнен|подтоп|затоп|вода зашла|вышла из берег|уровень воды|ледоход|затор|лесн\w* пожар|"
          r"природн\w* пожар|ландшафтн|(?<![а-яё])палы|задымлен|ураган|шквал|смерч|сильн\w* ветер|ледян\w* дожд|снегопад|"
          r"метел|гололед|обесточ|без света|без тепла|без электр|без воды|отключен|авари\w* на (?:сет|котельн|водопровод|тепло)|"
          r"котельн|эвакуир|отселен|разруш|обрушен|чрезвычайн\w* ситуац|режим\w* (?:чс|чрезвычайн)|повышенн\w* готовност|"
          r"беспилот|бпла|атак")
NOT_EVENT = (r"учени|тренировк|отработа|школ\w* безопасност|соревнован|конкурс|заняти|семинар|приложени|правила|памятк|"
             r"напомина|рекоменд|совещани|заседани")
CAUSES = ["паводок/наводнение", "лесные/природные пожары", "авария ЖКХ/тепло-, энерго-, водоснабжение",
          "метеоявления (ветер, снег, мороз)", "техногенная авария/разлив/обрушение/взрыв", "атаки БПЛА/обстрелы", "иное"]
# малая модель лучше отвечает на отдельные да/нет, чем выбирает из длинного списка (и тянется к первому варианту)
SCHEMA = {"type": "object", "properties": {
    "учения_проверка_подготовка_или_совещание": {"type": "boolean"},
    "прогноз_или_предупреждение_об_угрозе": {"type": "boolean"},
    "происшествие_с_отдельными_людьми_или_одним_домом": {"type": "boolean"},
    "массовое_событие_уже_происходит_на_территории": {"type": "boolean"},
    "причина": {"type": "string", "enum": CAUSES},
    "места": {"type": "array", "items": {"type": "string"}}},
    "required": ["учения_проверка_подготовка_или_совещание", "прогноз_или_предупреждение_об_угрозе",
                 "происшествие_с_отдельными_людьми_или_одним_домом", "массовое_событие_уже_происходит_на_территории",
                 "причина", "места"]}
PROMPT = """Заголовок новости регионального управления МЧС России. Ответь только по тексту заголовка, не используя свои
знания о событиях.
- учения_проверка_подготовка_или_совещание: учения, тренировки, проверки готовности, совещания, обучение, профилактика;
- прогноз_или_предупреждение_об_угрозе: ожидается, прогнозируется, возможна угроза, предупреждение;
- происшествие_с_отдельными_людьми_или_одним_домом: пожар в доме, ДТП, спасение туриста, утонул человек и т.п.;
- массовое_событие_уже_происходит_на_территории: уже подтоплены дома или дороги, горит лес у населённых пунктов,
  без тепла, света или воды остались жители, разрушения от стихии, эвакуированы жители, введён режим ЧС;
- причина; места — населённые пункты и районы, где по тексту происходит событие (если не названы — пустой список).

Примеры:
«Спасатели провели учения по ликвидации последствий паводка» → учения: да, массовое: нет.
«В выходные в регионе ожидается усиление ветра до 25 м/с» → прогноз: да, массовое: нет.
«Пожарные ликвидировали возгорание в частном доме в Ивановке» → происшествие: да, массовое: нет.
«В Приреченске продолжается откачка воды из подтопленных домов» → массовое: да, причина паводок, места [Приреченск].
«Более двух тысяч жителей посёлка Лесной остались без тепла после аварии на котельной» → массовое: да, причина ЖКХ.

ЗАГОЛОВОК: {t}"""


# После модели — правило: призывы, вопросы, мониторинг и предвестники (ледоход) без слов о последствиях — не событие.
APPEAL = r"[?!]|почему|как\s|мониторинг|контрол|противопаводков|профилакт|свидетел|особое внимание|ледоход|вскрыти|актуальное"
IMPACT = r"подтоп|затоп|эвакуир|режим\w* (?:чс|чрезвычайн)|пострадав|разруш|без тепла|без света|обесточ|ухудшил"
NEGATION = r"не произош|не зарегистр|не поступ|не зафикс|не выявл|нет подтоп|не допущ|не повлия|не пострадал|угроз\w* .{0,20}снят"
# плановые сводки половодья, мелкие палы травы, единичные пожары, онлайн-карты, ледовые переправы, «завершилась фаза»:
# без слов о последствиях для жителей это не событие
ROUTINE = (r"обстановк|информаци\w* о прохождении|данные по|итоги|по состоянию на|за сутки|за неделю|с начала|онлайн-карт|"
           r"ледов\w* переправ|завершил|встречает|палы|пал\w* трав|травян|ландшафтн\w* (?:пожар|возгоран)|возгоран|очаг|потушил|спасли|«")
STRONG = r"подтоплен\w* (?:дом|двор|сел|пос|улиц|населен|территор|микрорайон)|затоплен|эвакуир\w* (?:жител|населен)|режим\w* (?:чс|чрезвычайн)|пострадавш\w* от|без тепла|без света|отключени\w* (?:тепло|электр)|аварийн\w* отключ|разруш|доставлен\w* продукт|в \d+[-\s]?километров"


def is_event(lab):
    """Реальное массовое событие: модель отметила его и не отметила учения, прогноз или единичное происшествие,
    а заголовок не похож на призыв или сводку без последствий."""
    title = str(lab.get("title", "")).lower()
    appeal = ((re.search(APPEAL, title) and not re.search(IMPACT, title)) or re.search(NEGATION, title)
              or (re.search(ROUTINE, title) and not re.search(STRONG, title)))
    return bool(lab.get("массовое_событие_уже_происходит_на_территории")) and not (
        lab.get("учения_проверка_подготовка_или_совещание") or lab.get("прогноз_или_предупреждение_об_угрозе")
        or lab.get("происшествие_с_отдельными_людьми_или_одним_домом") or appeal)


def candidates():
    """Заголовки всех регионов, прошедшие словарный фильтр."""
    parts = []
    for p in sorted((MCHS_DIR / "titles").glob("titles_*.parquet")):
        t = pd.read_parquet(p)
        if len(t):
            has = lambda rx: t.title.str.contains(rx, case=False, regex=True)
            parts.append(t[has(HAZARD) & ~has(NOT_EVENT)])
    return pd.concat(parts, ignore_index=True)


def path():
    INTERIM.mkdir(parents=True, exist_ok=True)
    return INTERIM / "mchs_labels.jsonl"


def label_all(threads=3):
    """Размечает ещё не размеченные заголовки-кандидаты, дописывая в jsonl (прерванный прогон продолжается)."""
    done = {json.loads(l)["id"] for l in open(path())} if path().exists() else set()
    todo = candidates()
    todo = todo[~todo.id.astype(int).isin(done)]

    def one(r):
        try:
            return r, llm.ask(PROMPT.format(t=r.title), SCHEMA, num_ctx=2048, timeout=120)
        except Exception as e:                        # сбой модели — строка не пишется и размечается при следующем запуске
            print("модель не ответила:", r.region_code, r.id, str(e)[:80], flush=True)
            return r, None

    with open(path(), "a", encoding="utf-8") as fh, ThreadPoolExecutor(threads) as ex:
        for k, (r, lab) in enumerate(ex.map(one, todo.itertuples())):
            if lab is None:
                continue
            fh.write(json.dumps({"id": int(r.id), "region_code": int(r.region_code), "dt": r.dt, "title": r.title} | lab,
                                ensure_ascii=False) + "\n")
            fh.flush()
            if k % 200 == 0:
                print(f"размечено {k} / {len(todo)}", flush=True)
    return len(todo)


def labels():
    return pd.DataFrame([json.loads(l) for l in open(path())]) if path().exists() else pd.DataFrame()


def body_path(region_code, nid):
    return MCHS_DIR / "bodies" / f"{int(region_code):02d}_{int(nid)}.txt"


def fetch_bodies():
    """Тексты новостей, которые модель отметила событием (по заголовку): в тексте названы населённые пункты и районы,
    которых нет в заголовке, — это главное для привязки к МО. Последовательно, с паузой crawl.pause_s из
    configs/base.yaml, с кэшем на диске."""
    pause = config()["crawl"]["pause_s"]
    lab = labels()
    lab = lab[lab.apply(is_event, axis=1)]
    todo = [r for r in lab.itertuples() if not body_path(r.region_code, r.id).exists()]
    for k, r in enumerate(todo):
        try:
            body = text(r.region_code, r.id)
        except Exception as e:                        # сбой сети — пропуск, повторится при следующем запуске
            print("текст не получен", r.region_code, r.id, str(e)[:60], flush=True)
            continue
        f = body_path(r.region_code, r.id)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
        if k % 100 == 0:
            print(f"тексты {k} / {len(todo)}", flush=True)
        time.sleep(pause)
    return len(todo)


FLOOD_WORDS = r"паводк|половод|подтоп|затоп|уровень воды|дамб|вода зашла|вышла из берег|эвакуир|отселен"


def label_posts(posts, name="tg", threads=3, pattern=HAZARD, until=None):
    """Разметка постов Telegram той же схемой: текст поста вместо заголовка (первые 700 знаков — главное в посте
    сказано в начале, а длинный запрос в разы замедляет модель); отбор — словарь pattern и дата не позже until;
    дозапись в data/interim/{name}_labels.jsonl."""
    out = INTERIM / f"{name}_labels.jsonl"
    done = {json.loads(l)["post"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [p for p in posts if p["post"] not in done and p["text"] and (until is None or p["dt"][:10] <= until)
            and re.search(pattern, p["text"], re.I) and not re.search(NOT_EVENT, p["text"][:300], re.I)]

    def one(p):
        try:
            return p, llm.ask(PROMPT.replace("Заголовок новости", "Сообщение").replace("ЗАГОЛОВОК", "СООБЩЕНИЕ")
                              .replace("по тексту заголовка", "по тексту сообщения").format(t=p["text"][:700]),
                              SCHEMA, num_ctx=2048, timeout=180)
        except Exception as e:                        # сбой модели — пост размечается при следующем запуске
            print("модель не ответила:", p["post"], str(e)[:80], flush=True)
            return p, None

    with open(out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(threads) as ex:
        for k, (p, lab) in enumerate(ex.map(one, todo)):
            if lab is None:
                continue
            fh.write(json.dumps({"post": p["post"], "region_code": p["region_code"], "dt": p["dt"], "title": p["text"][:300]} | lab,
                                ensure_ascii=False) + "\n")
            fh.flush()
            if k % 100 == 0:
                print(f"посты: {k} / {len(todo)}", flush=True)
    return len(todo)
