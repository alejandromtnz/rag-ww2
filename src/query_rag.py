import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

# -------------------------
# Config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "index"

FAISS_PATH = INDEX_DIR / "faiss_index.bin"
META_PATH = INDEX_DIR / "metadatos.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index_and_metadata():
    print("[INFO] Cargando índice FAISS y metadatos...")
    index = faiss.read_index(str(FAISS_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] Chunks en metadatos: {len(metadata)}")
    return index, metadata


def embed_query(model, query: str):
    return model.encode([query])


def search(index, metadata, model, query: str, top_k: int = 5):
    # 1) Embedding de la pregunta
    q_vec = embed_query(model, query)

    # 2) Búsqueda en FAISS
    distances, indices = index.search(q_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        doc = metadata[idx]
        results.append(
            {
                "rank": rank + 1,
                "score": float(distances[0][rank]),
                "id": doc.get("id"),
                "fuente": doc.get("fuente"),
                "texto": doc.get("texto", "")[:500] + "..."  # recortamos un poco
            }
        )
    return results


def main():
    # Cargar modelo y datos
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index, metadata = load_index_and_metadata()

    print("\n[READY] Sistema RAG de prueba (sin LLM aún).")
    print("Escribe una pregunta sobre la Segunda Guerra Mundial o sobre países/regiones.")
    print("Escribe 'salir' para terminar.\n")

    while True:
        query = input("Tú: ").strip()
        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            print("Bye!")
            break

        results = search(index, metadata, model, query, top_k=5)

        print("\n[RESULTADOS]")
        if not results:
            print("No se encontraron chunks relevantes.")
        else:
            for r in results:
                print(f"\n--- Resultado {r['rank']} (score={r['score']:.4f}) ---")
                print(f"ID: {r['id']} | Fuente: {r['fuente']}")
                print(r["texto"])
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
