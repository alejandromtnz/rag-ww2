import json
from pathlib import Path

try:
    from config import DATA_PROCESSED, CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200


def load_jsonl(path: Path):
    """Carga documentos desde un archivo JSONL, con mensajes de depuración."""
    if not path.exists():
        print(f"[WARN] No se encontró: {path}")
        return []
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                docs.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] Línea {i} no válida en {path.name}: {e}")
    print(f"[INFO] Cargados {len(docs)} documentos desde {path.name}")
    if docs:
        # Mostrar un ejemplo
        ejemplo = docs[0]
        print(f"[DEBUG] Ejemplo de doc en {path.name}: keys={list(ejemplo.keys())}")
    return docs


def chunk_text(text: str, size: int, overlap: int):
    """Divide un texto largo en chunks solapados."""
    text = text or ""
    length = len(text)
    if length == 0:
        return []

    chunks = []
    start = 0

    while start < length:
        end = min(start + size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    wiki_path = DATA_PROCESSED / "wiki_docs.jsonl"
    geo_pdf_path = DATA_PROCESSED / "geo_pdf_docs.jsonl"

    # 1) Cargar fuentes
    wiki_docs = load_jsonl(wiki_path)
    geo_docs = load_jsonl(geo_pdf_path)

    all_docs = wiki_docs + geo_docs
    print(f"[INFO] Total documentos cargados (todas las fuentes): {len(all_docs)}")

    if len(all_docs) == 0:
        print("[ERROR] No se ha cargado NINGÚN documento.")
        print("-> Revisa que ingest_wikipedia.py y ingest_geo_pdf.py hayan generado contenido.")
        return

    final_path = DATA_PROCESSED / "documentos.jsonl"

    with open(final_path, "w", encoding="utf-8") as f_out:
        total_chunks = 0

        for idx, doc in enumerate(all_docs, start=1):
            texto = doc.get("texto", "")
            if not texto:
                print(f"[WARN] Doc {idx} sin campo 'texto' o vacío, id={doc.get('id')}")
                continue

            chunks = chunk_text(texto, CHUNK_SIZE, CHUNK_OVERLAP)
            if idx == 1:
                # debug solo para el primer doc
                print(f"[DEBUG] Doc {idx} (id={doc.get('id')}), longitud texto={len(texto)}, chunks generados={len(chunks)}")

            for i, ch in enumerate(chunks):
                new_doc = {
                    "id": f"{doc.get('id', 'doc')}_chunk{i}",
                    "texto": ch,
                    "fuente": doc.get("fuente", "desconocida"),
                    "metadata": doc.get("metadata", {})
                }
                f_out.write(json.dumps(new_doc, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[DONE] Dataset final guardado en: {final_path}")
    print(f"[DONE] Total de chunks generados: {total_chunks}")


if __name__ == "__main__":
    main()
