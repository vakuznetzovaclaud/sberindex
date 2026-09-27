"""Заголовки новостей ГУ МЧС по всем покрытым регионам за 2023–2024 (параллельно по регионам, с дозаписью).
Запуск: python scripts/build_mchs.py [число потоков]"""
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ews import news_mchs
from ews.paths import PROCESSED

OUT = news_mchs.DIR / "titles"


def one(code):
    f = OUT / f"titles_{code:02d}.parquet"
    if f.exists():
        return code, len(pd.read_parquet(f))
    parts = [news_mchs.listing(code, a, b) for a, b in [("01.01.2023", "30.06.2023"), ("01.07.2023", "31.12.2023"),
                                                         ("01.01.2024", "30.06.2024"), ("01.07.2024", "31.12.2024")]]
    df = pd.concat(parts).drop_duplicates("id")
    df.to_parquet(f)
    return code, len(df)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    codes = sorted(set(pd.read_parquet(PROCESSED / "mo_meta.parquet")["region_code"].astype(int)))
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        for code, n in ex.map(one, codes):
            print(f"регион {code:02d}: {n} новостей", flush=True)
