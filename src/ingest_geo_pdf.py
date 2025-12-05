from pathlib import Path
import json

try:
    # Si ya tienes un config.py parecido al del proyecto anterior
    from config import DATA_RAW, DATA_PROCESSED
    PDF_DIR = DATA_RAW / "pdfs"
except ImportError:
    # Fallback: rutas por defecto
    BASE_DIR = Path(__file__).resolve().parent.parent
    PDF_DIR = BASE_DIR / "data" / "raw" / "pdfs"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"

from PyPDF2 import PdfReader  # pip install PyPDF2

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    pdf_path = PDF_DIR / "country-regions.pdf"  # nombre del PDF que has guardado
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se ha encontrado el PDF en: {pdf_path}")

    print(f"[INFO] Extrayendo texto de: {pdf_path}")
    texto = extract_text_from_pdf(pdf_path)

    doc = {
        "id": "geo_country_regions",
        "texto": texto,
        "fuente": "geografia_pdf",
        "metadata": {
            "filename": pdf_path.name,
            "source": "IOM / UN M49 country regions",
            "url": "https://data.iom.int/codelist/country-regions.pdf"
        }
    }

    out_path = DATA_PROCESSED / "geo_pdf_docs.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[DONE] Documento guardado en: {out_path}")

if __name__ == "__main__":
    main()
