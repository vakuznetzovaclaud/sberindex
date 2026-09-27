"""Реестр актов о ЧС и повышенной готовности 2023–2024: список, OCR, извлечение фактов (параллельно, с дозаписью).
Запуск: python scripts/build_acts.py [ocr|extract|all] [число потоков извлечения]"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pandas as pd

from ews import acts
from ews.paths import PROCESSED


def selected():
    f = acts.DIR / "acts_2023_2024.parquet"
    df = pd.read_parquet(f) if f.exists() else acts.fetch_list()
    covered = set(pd.read_parquet(PROCESSED / "mo_meta.parquet")["region_code"].astype(int))
    sel = df[df.region_code.isin(covered) & df.action.isin(["введение", "изменение", "иное"])]
    return sel.sort_values("pub_date", ascending=False).reset_index(drop=True)   # сначала 2024 г.: для него есть отклик трат h = 1


def _ocr_one(eo):
    try:
        acts.text_of(eo)
        return eo, None
    except Exception as e:
        return eo, str(e)[:120]


def do_ocr(df, workers=4):
    with ProcessPoolExecutor(workers) as ex:
        for i, (eo, err) in enumerate(ex.map(_ocr_one, df.eo)):
            if err:
                print("OCR не удалось", eo, err, flush=True)
            if i % 25 == 0:
                print("OCR", i, "/", len(df), flush=True)


def do_extract(df, workers=3):
    out = acts.extracted_path()
    done = {json.loads(l)["eo"] for l in open(out)} if out.exists() else set()
    todo = [eo for eo in df.eo if eo not in done and (acts.DIR / "ocr" / f"{eo}.txt").exists()]
    print("к извлечению:", len(todo), flush=True)

    def one(eo):
        try:
            return {"eo": eo} | acts.extract(eo)
        except Exception as e:
            return {"eo": eo, "ошибка": str(e)[:200]}

    with ThreadPoolExecutor(workers) as ex, open(out, "a", encoding="utf-8") as fh:
        for i, r in enumerate(ex.map(one, todo)):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
            if i % 25 == 0:
                print("извлечено", i, "/", len(todo), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    df = selected()
    print("актов в покрытых регионах:", len(df), flush=True)
    if stage in ("ocr", "all"):
        for i, eo in enumerate(df.eo):                 # загрузка — последовательно, портал режет параллельные запросы
            try:
                acts.pdf_of(eo)
            except Exception as e:
                print("PDF не получен", eo, str(e)[:80], flush=True)
            if i % 50 == 0:
                print("PDF", i, "/", len(df), flush=True)
        do_ocr(df[[(acts.DIR / "pdf" / f"{eo}.pdf").exists() for eo in df.eo]])
    if stage in ("extract", "all"):
        do_extract(df, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
