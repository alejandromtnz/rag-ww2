import sys
import os
import pathlib
import time
import streamlit as st

# =========================================
# IMPORTS DE RUTAS
# =========================================
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from rag_chat import answer_with_rag


# =========================================
# CONFIG STREAMLIT + FUENTES
# =========================================
st.set_page_config(
    page_title="RAG WW2 Chatbot",
    page_icon="🪖",
    layout="wide"
)

# ======================
# ESTILO Y TIPO DE LETRA
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

html, body, * {
    font-family: 'Poppins', sans-serif !important;
    color: white;
}

[data-testid="stAppViewContainer"] {
    background-color: #0c0f16;
}

[data-testid="stSidebar"] {
    background-color: #141720;
}

.chat-bubble-user {
    background-color: #2b2d36;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 6px;
}

.chat-bubble-assistant {
    background-color: #1f4a72;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 6px;
}

hr {
    border: none;
    height: 1px;
    background: #374151;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# SIDEBAR
# =========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/World_War_II_Montage.png/440px-World_War_II_Montage.png")

    st.markdown("### 🧠 Sistema RAG para WW2")
    st.markdown("""
    Este chatbot utiliza:
    - FAISS como motor principal
    - Embeddings MiniLM
    - Llama 3.1 vía Ollama
    - Dataset histórico indexado
    """)

    st.markdown("---")

    if st.button("🗑 Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()


# =========================================
# CABECERA PRINCIPAL
# =========================================
st.markdown("<h1 style='text-align: center;'>Chatbot RAG WWII</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Haz preguntas con rigor histórico basado SOLO en documentos indexados.</p>", unsafe_allow_html=True)


# =========================================
# HISTORIAL
# =========================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# Mostrar mensajes previos
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-assistant'>🎖️ {msg['content']}</div>", unsafe_allow_html=True)


# =========================================
# INPUT DEL USUARIO
# =========================================
question = st.chat_input("Pregunta sobre batallas, líderes, fechas o hechos históricos...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(f"<div class='chat-bubble-user'>🧑 {question}</div>", unsafe_allow_html=True)

    with st.spinner("Buscando información real y contrastada..."):
        response = answer_with_rag(question)
        answer = response["answer"]

        # Efecto typing
        placeholder = st.empty()
        typed = ""
        for ch in answer:
            typed += ch
            placeholder.markdown(f"<div class='chat-bubble-assistant'>🎖️ {typed}█</div>", unsafe_allow_html=True)
            time.sleep(0.008)
        placeholder.markdown(f"<div class='chat-bubble-assistant'>🎖️ {answer}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # =========================================
    # CONTEXTO UTILIZADO
    # =========================================
    with st.expander("📄 Documentos utilizados para responder"):
        for i, doc in enumerate(response.get("context_docs", []), start=1):
            st.markdown(f"### 📌 Documento nº {i}")
            fuente = doc.get("fuente", "?")
            meta = doc.get("metadata", {})
            title = meta.get("title") or meta.get("filename") or "Sin título"
            st.markdown(f"**Fuente:** `{fuente}`")
            st.markdown(f"**Título/Origen:** `{title}`")

            # Mostrar extracto real
            texto = doc.get("texto", "")
            st.code(texto[:800] + ("..." if len(texto) > 800 else ""), language="markdown")

            st.markdown("---")
