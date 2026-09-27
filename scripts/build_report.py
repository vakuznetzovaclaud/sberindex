"""Сборка методологического отчёта: report/methodology.md в HTML и PDF (печать headless Chrome/Chromium).
Рисунки подключаются относительными путями из report/fig, поэтому HTML открывается и сам по себе.
Запуск: python scripts/build_report.py"""
import subprocess
import sys
from pathlib import Path

import markdown

from ews.web import chrome

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
TITLE = "Где искать шок: методологический отчёт"

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm; }
:root { --ink: #16181d; --muted: #5b6270; --rule: #d9dde3; --accent: #0b6e4f; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.3pt; line-height: 1.47; color: var(--ink);
       max-width: 178mm; margin: 0 auto; }
h1 { font-size: 19pt; line-height: 1.22; margin: 0 0 4pt; letter-spacing: -0.2pt; }
h1 + p { color: var(--muted); margin-bottom: 14pt; }
h2 { font-size: 13.2pt; margin: 20pt 0 6pt; padding-top: 6pt; border-top: 1.2pt solid var(--ink); page-break-after: avoid; }
h3 { font-size: 11pt; margin: 13pt 0 4pt; page-break-after: avoid; }
p { margin: 5pt 0; }
strong { font-weight: 650; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 7pt 0 9pt; page-break-inside: avoid; }
th, td { border-bottom: 0.6pt solid var(--rule); padding: 2.6pt 4pt; text-align: left; vertical-align: top; }
th { border-bottom: 1pt solid var(--ink); font-weight: 650; }
td:not(:first-child), th:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; }
ul, ol { margin: 4pt 0; padding-left: 15pt; }
li { margin: 2pt 0; }
code { font-family: Menlo, monospace; font-size: 8.8pt; background: #f3f4f6; padding: 0 2px; border-radius: 2px; }
img { max-width: 100%; display: block; margin: 9pt auto 3pt; page-break-inside: avoid; }
p > em:only-child { display: block; font-size: 8.8pt; color: var(--muted); margin: 0 0 9pt; }
a { color: var(--accent); text-decoration: none; }
"""


def main():
    body = markdown.markdown((REPORT / "methodology.md").read_text(encoding="utf-8"), extensions=["tables", "attr_list", "toc"])
    html = f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{TITLE}</title><style>{CSS}</style></head><body>{body}</body></html>'
    out = REPORT / "methodology.html"
    out.write_text(html, encoding="utf-8")
    try:
        exe = chrome()
    except RuntimeError as e:
        print(e, "— собран только HTML:", out)
        return
    pdf = REPORT / "methodology.pdf"
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                    f"--print-to-pdf={pdf}", out.as_uri()], check=True, capture_output=True)
    from pypdf import PdfReader, PdfWriter
    r, w = PdfReader(pdf), PdfWriter()
    for page in r.pages:
        w.add_page(page)
    w.add_metadata({"/Title": TITLE, "/Subject": "Онлайн-конкурс СберИндекса 2026: прогнозирование и обнаружение точек структурных изменений"})
    with open(pdf, "wb") as fh:
        w.write(fh)
    print(f"{pdf.relative_to(ROOT)}: страниц {len(r.pages)}, {pdf.stat().st_size / 1e6:.2f} МБ")


if __name__ == "__main__":
    sys.exit(main())
