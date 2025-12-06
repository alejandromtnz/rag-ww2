
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


st.title("Chatbot RAG – WW2")
st.caption("Pregunta lo que quieras sobre la Segunda Guerra Mundial. Respuestas basadas SOLO en tu dataset indexado.")


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

question = st.chat_input("Haz tu pregunta aquí...")

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
    with st.expander("Documentos usados"):

        # CSS para hacer el texto más pequeño y la cita estilizada
        st.markdown(
            """
            <style>
            .small-text {
                font-size: 0.85rem;
            }
            .quote-box {
                font-size: 0.80rem;
                font-style: italic;
                color: #555;
                padding-left: 10px;
                border-left: 3px solid #ccc;
                margin-top: 4px;
                margin-bottom: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Mostrar solo 2 documentos
        for i, d in enumerate(result.get("context_docs", [])[:2], start=1):
            fuente = d.get("fuente", "")
            meta = d.get("metadata", {}) or {}
            title = meta.get("title") or meta.get("filename") or ""

            # Intentar varios nombres de campo para el texto del chunk
            raw_text = (
                d.get("content")
                or d.get("text")
                or d.get("page_content")
                or d.get("chunk")
                or d.get("body")
                or d.get("texto")
                or ""
            )

            # Si aun así está vacío, ponemos un mensaje por defecto
            if raw_text.strip() == "":
                snippet = "Fragmento no disponible en los metadatos del documento."
            else:
                snippet = (raw_text[:200] + "…") if len(raw_text) > 200 else raw_text

            # Mostrar encabezado
            st.markdown(
                f"<div class='small-text'><strong>Documento {i}</strong></div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='small-text'>Fuente: {fuente}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='small-text'>Título: {title}</div>",
                unsafe_allow_html=True
            )

            # Mostrar cita
            st.markdown(
                f"<div class='quote-box'>“{snippet}”</div>",
                unsafe_allow_html=True
            )

            st.markdown("<hr>", unsafe_allow_html=True)


