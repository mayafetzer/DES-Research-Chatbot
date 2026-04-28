import streamlit as st
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ================================
# DEVICE
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"

st.title("📄 RAG PDF Chatbot (Fixed Version)")

# ================================
# SESSION STATE (IMPORTANT FIX)
# ================================
if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================================
# CACHE MODELS
# ================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_llm():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    return tokenizer, model


embedder = load_embedder()
tokenizer, model = load_llm()

# ================================
# PDF FUNCTIONS
# ================================
def extract_text(files):
    text = ""
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+size])
        start += size - overlap
    return chunks


def build_index(chunks):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index


# ================================
# UPLOAD SECTION
# ================================
st.sidebar.header("Upload PDFs")
files = st.sidebar.file_uploader("Upload", type=["pdf"], accept_multiple_files=True)

if files and st.sidebar.button("Build Knowledge Base"):
    with st.spinner("Processing PDFs..."):
        text = extract_text(files)

        if len(text.strip()) == 0:
            st.error("No text found in PDFs.")
            st.stop()

        chunks = chunk_text(text)

        if len(chunks) == 0:
            st.error("Chunking failed.")
            st.stop()

        index = build_index(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("Knowledge base ready!")

# ================================
# RETRIEVAL
# ================================
def retrieve(query):
    query_vec = embedder.encode([query])
    D, I = st.session_state.index.search(np.array(query_vec).astype("float32"), 3)
    return [st.session_state.chunks[i] for i in I[0]]


# ================================
# CHAT FUNCTION
# ================================
def ask_llm(query):
    context = "\n\n".join(retrieve(query))

    prompt = f"""
You are a research assistant.

Use ONLY context below:

{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()


# ================================
# CHAT UI
# ================================
st.subheader("Chat")

query = st.text_input("Ask something from your PDFs")

if query:
    if st.session_state.index is None:
        st.warning("Upload PDFs first.")
    else:
        answer = ask_llm(query)
        st.session_state.chat.append((query, answer))


# ================================
# DISPLAY CHAT
# ================================
for q, a in reversed(st.session_state.chat):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
