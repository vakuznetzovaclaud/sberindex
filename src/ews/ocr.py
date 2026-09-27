"""Распознавание сканов PDF: страницы рендерит pdftoppm (poppler), текст — Apple Vision (ocrmac) на macOS или Tesseract
(rus) в других системах. Результаты OCR сохраняются, поэтому повторный прогон распознавание не требует."""
import shutil
import subprocess
import tempfile
from pathlib import Path


def pdf_to_text(pdf: Path, dpi=200, max_pages=8) -> str:
    """Первые max_pages страниц: причина, даты и территория режима — в начале акта; дальше обычно приложения-перечни."""
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-l", str(max_pages), "-png", str(pdf), f"{td}/p"], check=True)
        pages = sorted(Path(td).glob("p-*.png"))
        return "\n".join(_image_to_text(p) for p in pages)


def _image_to_text(png: Path) -> str:
    try:
        from ocrmac import ocrmac
        return "\n".join(a[0] for a in ocrmac.OCR(str(png), language_preference=["ru-RU"], recognition_level="accurate").recognize())
    except ImportError:
        if not shutil.which("tesseract"):
            raise RuntimeError("нужен ocrmac (macOS) или tesseract с языком rus")
        return subprocess.run(["tesseract", str(png), "-", "-l", "rus"], capture_output=True, text=True).stdout
