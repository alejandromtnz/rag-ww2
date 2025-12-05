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
            textos.append(d["texto"])
            metadatos.append(d)

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
