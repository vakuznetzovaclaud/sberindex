"""PDF-версии: страница site/index.html в site/landing.pdf, презентация site/presentation.html в site/presentation.pdf
(Chrome без окна).
Ссылки на локальные файлы (file://…) Chrome вписывает в PDF с полным путём — они удаляются, внешние остаются.
Запуск: python scripts/print_landing.py  (путь к Chrome — переменная CHROME, если он не в стандартном месте)"""
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject

from ews.web import chrome

SITE = Path(__file__).resolve().parents[1] / "site"


def to_pdf(html, pdf):
    raw = Path(tempfile.mkdtemp()) / "out.pdf"
    subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--no-sandbox", "--virtual-time-budget=20000",
                    "--no-pdf-header-footer", f"--print-to-pdf={raw}", html.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    reader, writer = PdfReader(raw), PdfWriter()
    dropped = 0
    for page in reader.pages:
        annots = page.get("/Annots")
        if annots:
            keep = []
            for a in annots:
                uri = a.get_object().get("/A", {}).get("/URI", "")
                if str(uri).startswith("file:"):
                    dropped += 1
                else:
                    keep.append(a)
            page[NameObject("/Annots")] = ArrayObject(keep)
        writer.add_page(page)
    writer.add_metadata({"/Title": reader.metadata.get("/Title", ""), "/Creator": "", "/Producer": ""})
    with open(pdf, "wb") as f:
        writer.write(f)
    print(f"{pdf.relative_to(SITE.parent)}: страниц {len(reader.pages)}, удалено локальных ссылок {dropped}")


def main():
    to_pdf(SITE / "index.html", SITE / "landing.pdf")
    to_pdf(SITE / "presentation.html", SITE / "presentation.pdf")


if __name__ == "__main__":
    main()
