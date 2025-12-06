
import sys
import os
import pathlib

# Ruta al directorio raíz del proyecto (rag_ww2/)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Ruta al directorio /src
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from rag_chat import answer_with_rag
# O alternativamente:
# from src.rag_chat import answer_with_rag

import streamlit as st


# ==========================
# CONFIG STREAMLIT
# ==========================

st.set_page_config(
    page_title="RAG WW2 Chatbot",
    page_icon="🪖",
)


st.title("Chatbot RAG – Segunda Guerra Mundial")
st.caption("Pregunta lo que quieras. Respuestas basadas SOLO en tu dataset indexado.")


# ==========================
# HISTORIAL DE CONVERSACIÓN
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================
# INPUT DEL USUARIO
# ==========================

question = st.chat_input("Haz tu pregunta sobre WW2...")

if question:
    # Añadir el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Llamada al RAG de verdad ❤️‍🔥
    with st.chat_message("assistant"):
        with st.spinner("Buscando información real y contrastada..."):
            result = answer_with_rag(question)  # Llama a tu backend real
            answer = result["answer"]

            st.markdown(answer)

    # Guardar en historial
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Mostrar contexto usado (opcional desplegable)
    with st.expander("📄 Documentos usados"):
        for i, d in enumerate(result.get("context_docs", []), start=1):
            fuente = d.get("fuente", "")
            meta = d.get("metadata", {})
            title = meta.get("title") or meta.get("filename") or ""

            st.markdown(f"### Documento {i}")
            st.markdown(f"**Fuente:** {fuente}")
            st.markdown(f"**Título:** {title}")
            st.markdown("---")