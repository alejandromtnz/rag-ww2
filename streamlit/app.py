import sys
import os
import pathlib
import streamlit as st

# =========================================
# RUTAS E IMPORTS
# =========================================
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

from rag_chat import answer_with_rag  # tu backend RAG


# =========================================
# CONFIG STREAMLIT + ESTILO GLOBAL
# =========================================
st.set_page_config(
    page_title="RAG WW2 Chatbot",
    page_icon="🪖",
    layout="wide"
)

# CSS: fuente, fondo, burbujas, tamaño docs
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* Fondo general */
[data-testid="stAppViewContainer"] {
    background-color: #0c0f16;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #141720;
}

/* Burbujas chat */
.chat-bubble-user {
    background-color: #2b2d36;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 6px;
}

.chat-bubble-assistant {
    background-color: #1f4a72;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 6px;
}

/* Código de documentos más pequeño */
pre, code {
    font-size: 0.70rem !important;
}

/* Título grande inicial */
.title-big {
    text-align: center;
    font-size: 2.5rem;
    margin-top: 0.5rem;
}

/* Título pequeño después de empezar a chatear */
.title-small {
    text-align: left;
    font-size: 1.5rem;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 0.95rem;
}

/* Lista de chats en sidebar */
.chat-name {
    font-size: 0.85rem;
    margin-bottom: 0.2rem;
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)


# =========================================
# ESTADO: MÚLTIPLES CHATS + DOCS
# =========================================
if "chats" not in st.session_state:
    st.session_state.chats = [{
        "id": 1,
        "name": "Chat 1",
        "messages": []
    }]
    st.session_state.current_chat_idx = 0

if "last_context_docs" not in st.session_state:
    st.session_state.last_context_docs = []

chats = st.session_state.chats
current_idx = st.session_state.current_chat_idx
current_chat = chats[current_idx]
messages = current_chat["messages"]
has_messages = len(messages) > 0


# =========================================
# SIDEBAR: INFO + LISTA DE CHATS
# =========================================
with st.sidebar:
    # Imagen (si no carga, no pasa nada)
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/"
            "World_War_II_Montage_A.jpg/640px-World_War_II_Montage_A.jpg",
            use_container_width=True
        )
    except Exception:
        pass

    st.markdown("### 🧠 Sistema RAG para WW2")
    st.markdown("""
    Este chatbot utiliza:
    - FAISS como motor principal  
    - Embeddings MiniLM  
    - Llama 3.1 vía Ollama  
    - Dataset histórico indexado  
    """)

    st.markdown("---")
    st.markdown("#### 💬 Chats guardados")

    for i, chat in enumerate(chats):
        label = chat["name"] or f"Chat {chat['id']}"
        if st.button(("👉 " if i == current_idx else "") + label, key=f"chat_btn_{i}"):
            st.session_state.current_chat_idx = i
            st.rerun()


# =========================================
# CABECERA (según si ya hay mensajes)
# =========================================
if not has_messages:
    st.markdown("<h1 class='title-big'>🪖 Chatbot RAG WWII</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Haz preguntas con rigor histórico basado SOLO en documentos indexados.</p>",
        unsafe_allow_html=True
    )
else:
    st.markdown("<h1 class='title-small'>🪖 Chatbot RAG WWII</h1>", unsafe_allow_html=True)

st.markdown("")  # pequeño espacio


# =========================================
# CONTENEDOR DEL CHAT (CONVERSACIÓN + DOCS)
# =========================================
chat_container = st.container()

with chat_container:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'>🎖️ {msg['content']}</div>", unsafe_allow_html=True)

    # documentos de la última respuesta (si los hay)
    docs = st.session_state.last_context_docs
    if docs:
        with st.expander("📄 Documentos utilizados para responder"):
            for i, doc in enumerate(docs, start=1):  # solo 2 docs ya recortados antes
                st.markdown(f"### 📌 Documento {i}")
                fuente = doc.get("fuente", "?")
                meta = doc.get("metadata", {})
                title = meta.get("title") or meta.get("filename") or "Sin título"

                st.markdown(f"**Fuente:** `{fuente}`")
                st.markdown(f"**Título/Origen:** `{title}`")

                texto = doc.get("texto", "") or ""
                snippet = texto[:800] + ("..." if len(texto) > 800 else "")
                st.code(snippet, language="markdown")
                st.markdown("---")


# =========================================
# INPUT SIEMPRE ABAJO + BOTONES
# =========================================
# 👉 Esto se queda SIEMPRE abajo, como en ChatGPT
question = st.chat_input(
    "Pregunta sobre batallas, líderes, fechas o hechos históricos...",
    key="chat_input_main"
)

# Estos botones irán justo encima de la barra de chat (no en la misma línea,
# pero muy cerca; con Streamlit puro no se puede meterlos dentro del chat_input)
col_in2, col_in3 = st.columns([1, 1])

with col_in2:
    clear_clicked = st.button("🗑 Limpiar", use_container_width=True)

with col_in3:
    new_chat_clicked = st.button("➕ Nuevo chat", use_container_width=True)


# manejar botones
if clear_clicked:
    current_chat["messages"] = []
    st.session_state.last_context_docs = []
    st.rerun()

if new_chat_clicked:
    new_id = max(c["id"] for c in chats) + 1 if chats else 1
    chats.append({
        "id": new_id,
        "name": f"Chat {new_id}",
        "messages": []
    })
    st.session_state.current_chat_idx = len(chats) - 1
    st.session_state.last_context_docs = []
    st.rerun()


# =========================================
# LÓGICA DE PREGUNTA / RESPUESTA
# (actualiza estado, pero NO pinta nada todavía)
# =========================================
if question:
    # actualizar nombre del chat con la primera pregunta
    if len(messages) == 0:
        snippet = question.strip()
        if len(snippet) > 30:
            snippet = snippet[:30] + "..."
        current_chat["name"] = snippet

    # guardar pregunta
    messages.append({"role": "user", "content": question})

    with st.spinner("Buscando información real y contrastada..."):
        result = answer_with_rag(question)
        answer = result["answer"]

    # guardar respuesta
    messages.append({"role": "assistant", "content": answer})

    # decidir si mostrar docs y guardarlos en estado
    answer_lower = answer.lower()
    context_docs = result.get("context_docs", []) or []

    show_docs = (
        context_docs
        and "no aparece en los documentos" not in answer_lower
        and "no aparece en el contexto" not in answer_lower
    )
    st.session_state.last_context_docs = context_docs[:2] if show_docs else []


# =========================================
# PINTAR CONVERSACIÓN + DOCS EN EL CONTENEDOR
# (aparece por encima del input)
# =========================================
with chat_container:
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'>🎖️ {msg['content']}</div>", unsafe_allow_html=True)

    # documentos de la última respuesta (si los hay)
    docs = st.session_state.last_context_docs
    if docs:
        with st.expander("📄 Documentos utilizados para responder"):
            for i, doc in enumerate(docs, start=1):  # solo 2 docs ya recortados antes
                st.markdown(f"### 📌 Documento {i}")
                fuente = doc.get("fuente", "?")
                meta = doc.get("metadata", {})
                title = meta.get("title") or meta.get("filename") or "Sin título"

                st.markdown(f"**Fuente:** `{fuente}`")
                st.markdown(f"**Título/Origen:** `{title}`")

                texto = doc.get("texto", "") or ""
                snippet = texto[:800] + ("..." if len(texto) > 800 else "")
                st.code(snippet, language="markdown")
                st.markdown("---")
