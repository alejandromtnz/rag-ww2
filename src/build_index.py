import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Intentamos usar config.py si existe
try:
    from config import DATA_PROCESSED, INDEX_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    INDEX_DIR = BASE_DIR / "index"

DOCUMENTS_FILE = DATA_PROCESSED / "documentos.jsonl"


def split_text(text: str, max_chars: int = 1200, overlap: int = 200):
    """
    Divide un texto largo en chunks de tamaño max_chars,
    solapados 'overlap' caracteres para no cortar ideas a la mitad.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar documentos (chunks)
    textos = []
    metadatos = []

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(f"No se encuentra {DOCUMENTS_FILE}")

    print(f"[INFO] Cargando documentos desde: {DOCUMENTS_FILE}")
    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texto_original = d.get("texto", "")
            if not texto_original:
                continue

            # Trocear el texto original en chunks más pequeños
            chunks = split_text(texto_original, max_chars=1200, overlap=200)

            for i, chunk in enumerate(chunks):
                nuevo_meta = dict(d)          # copia superficial
                nuevo_meta["texto"] = chunk   # sustituimos por el trozo
                nuevo_meta["chunk_id"] = i    # opcional: índice del chunk

                textos.append(chunk)
                metadatos.append(nuevo_meta)

    print(f"[INFO] Total de chunks cargados: {len(textos)}")

    # 2. Cargar modelo de embeddings (gratuito, local)
    print("[INFO] Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Calcular embeddings
    print("[INFO] Calculando embeddings...")
    embeddings = model.encode(textos, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # 4. Crear índice FAISS
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"[INFO] Índice FAISS creado con {index.ntotal} vectores.")

    # 5. Guardar índice y metadatos
    index_path = INDEX_DIR / "faiss_index.bin"
    meta_path = INDEX_DIR / "metadatos.json"

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False)

    print(f"[DONE] Índice guardado en: {index_path}")
    print(f"[DONE] Metadatos guardados en: {meta_path}")


if __name__ == "__main__":
    main()
