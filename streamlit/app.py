import sys
import os
import pathlib
import streamlit as st

# Ruta al directorio raíz del proyecto (rag_ww2/)
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Ruta al directorio /src
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from rag_chat import answer_with_rag


# ==========================
# CONFIG STREAMLIT
# ==========================

st.set_page_config(
    page_title="RAG WW2 Chatbot",
    page_icon="🪖",
)

# ==========================
# ESTILO GLOBAL
# ==========================

st.markdown("""
<style>

/* Tipografía global super compatible */
html, body, [class*="css"] {
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif !important;
}

/* Centrar título */
.centered-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 700;
    margin-top: -10px;
    margin-bottom: 5px;
}

/* Subtítulo centrado */
.centered-subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)



# ==========================
# TÍTULO Y SUBTÍTULO
# ==========================

st.markdown("<h1 class='centered-title'>Chatbot WW2 - RAG</h1>", unsafe_allow_html=True)
st.markdown("<p class='centered-subtitle'>Pregunta lo que quieras sobre la Segunda Guerra Mundial. Respuestas basadas SOLO en dataset indexado.</p>", unsafe_allow_html=True)


# ==========================
# HISTORIAL DE CONVERSACIÓN
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================
# INPUT DEL USUARIO
# ==========================

question = st.chat_input("Haz tu pregunta aquí...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando información real y contrastada..."):
            result = answer_with_rag(question)
            answer = result["answer"]

            st.markdown(answer)

            # Detectar si no hay información y extraer documentos
            context_docs = result.get("context_docs", [])
            no_info = answer.startswith((
                "No hay información",
                "No aparece información",
                "No se ha encontrado información",
                "No existe información",
                "No dispongo de información",
                "no hay información",
                "no aparece información",
                "no se ha encontrado información",
                "no existe información",
                "no dispongo de información",
                "No aparece en los documentos",
                "no aparece información"
            ))

    st.session_state.messages.append({"role": "assistant", "content": answer})


    # ==========================
    # DOCUMENTOS USADOS
    # ==========================

    if context_docs and not no_info:
        with st.expander("📄 Documentos usados"):

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

            for i, d in enumerate(context_docs[:2], start=1):
                fuente = d.get("fuente", "")
                meta = d.get("metadata", {}) or {}
                title = meta.get("title") or meta.get("filename") or ""

                raw_text = (
                    d.get("content")
                    or d.get("text")
                    or d.get("page_content")
                    or d.get("chunk")
                    or d.get("body")
                    or d.get("texto")
                    or ""
                )

                if raw_text.strip() == "":
                    snippet = "Fragmento no disponible en los metadatos del documento."
                else:
                    snippet = (raw_text[:200] + "…") if len(raw_text) > 200 else raw_text

                st.markdown(f"<div class='small-text'><strong>Documento {i}</strong></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-text'>Fuente: {fuente}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='small-text'>Título: {title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='quote-box'>“{snippet}”</div>", unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
