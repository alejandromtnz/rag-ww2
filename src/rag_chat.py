import json
from pathlib import Path
from typing import List, Dict, Any
import os

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ==========================
# RUTAS Y CONFIGURACIÓN
# ==========================

try:
    from config import INDEX_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent
    INDEX_DIR = BASE_DIR / "index"

INDEX_PATH = INDEX_DIR / "faiss_index.bin"
META_PATH = INDEX_DIR / "metadatos.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMA_MODEL = "llama3.1:8b"


# ==========================
# CARGA DE ÍNDICE Y MODELO
# ==========================

print("[INFO] Cargando índice FAISS y metadatos...")
if not INDEX_PATH.exists():
    raise FileNotFoundError(f"No se encuentra el índice: {INDEX_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"No se encuentran los metadatos: {META_PATH}")

index = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "r", encoding="utf-8") as f:
    METADATOS: List[Dict[str, Any]] = json.load(f)

print(f"[INFO] Chunks en índice: {index.ntotal}")
print("[INFO] Cargando modelo de embeddings...")
os.environ["HF_HUB_OFFLINE"] = "1"
# embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True) # para local
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ==========================
# RETRIEVAL
# ==========================

def retrieve_context(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Dada una pregunta, devuelve los k chunks más parecidos.
    """
    q_vec = embedder.encode([question]).astype("float32")
    distances, indices = index.search(q_vec, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(METADATOS):
            results.append(METADATOS[idx])
    return results


# ==========================
# LLAMADA A LLAMA (OLLAMA)
# ==========================

def call_llama(prompt: str, system_prompt: str | None = None) -> str:
    """
    Llama al modelo llama3.1:8b a través de Ollama.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Formato típico de Ollama
    if isinstance(data, dict) and "message" in data:
        return data["message"]["content"]

    return str(data)


# ==========================
# CONSTRUCCIÓN DEL PROMPT RAG
# ==========================

def build_rag_prompt(question: str, context_docs: List[Dict[str, Any]]) -> str:
    """
    Construye el prompt que se le pasa a Llama usando los textos recuperados.
    """
    context_parts = []
    for doc in context_docs:
        fuente = doc.get("fuente", "desconocida")
        meta = doc.get("metadata", {}) or {}
        title = meta.get("title") or meta.get("filename") or ""
        prefix = f"[Fuente: {fuente}"
        if title:
            prefix += f" | Título: {title}"
        prefix += "]\n"
        context_parts.append(prefix + doc.get("texto", ""))

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""
Usa EXCLUSIVAMENTE la siguiente información de contexto para responder a la pregunta.
Si la respuesta no está claramente en el contexto, di que no aparece en los documentos.

Si la pregunta pide un número (personas, muertos, años, fechas), responde
PRIMERO con la cifra o la fecha, y luego añade como máximo una breve explicación
de 1–2 frases. No des contexto histórico general si no es necesario o no se pide.

Contexto:
{context_str}

Pregunta del usuario:
{question}

Responde en español, de forma clara y breve (máximo 2 párrafos de 2 o 3 frases cada uno).
"""
    return prompt.strip()


# ==========================
# FUNCIÓN PRINCIPAL RAG
# ==========================

def answer_with_rag(question: str, k: int = 5) -> Dict[str, Any]:
    """
    Recupera contexto + genera respuesta con Llama.
    """
    context_docs = retrieve_context(question, k=k)
    prompt = build_rag_prompt(question, context_docs)

    system_prompt = (
        "Eres un asistente experto en Segunda Guerra Mundial y sabes mucho sobre geografía mundial. "
        "Tu prioridad es responder de forma directa y concisa a la pregunta concreta del usuario. "
        "No te extiendas con contexto histórico general si no es necesario. "
        "Respondes SIEMPRE en español, usando solo la información del contexto que te doy. "
        "Si el contexto no tiene la respuesta, dilo claramente sin inventar."
    )

    answer = call_llama(prompt, system_prompt=system_prompt)

    return {
        "question": question,
        "answer": answer,
        "context_docs": context_docs,
    }


# ==========================
# CLI DE PRUEBA
# ==========================

if __name__ == "__main__":
    print(">>> Chat RAG con Llama 3.1:8B (Ollama). Escribe 'salir' para terminar.")
    while True:
        q = input("\nTú: ").strip()
        if not q:
            continue
        if q.lower() in {"salir", "exit", "quit"}:
            break

        result = answer_with_rag(q, k=5)

        print("\nAsistente:\n")
        print(result["answer"])

        print("\n[Fuentes usadas]")
        for i, doc in enumerate(result["context_docs"], start=1):
            fuente = doc.get("fuente", "desconocida")
            meta = doc.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("filename") or ""
            print(f"{i}. {fuente} - {title}")