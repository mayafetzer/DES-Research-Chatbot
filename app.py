import streamlit as st
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ================================
# LOAD MODELS (cache for speed)
# ================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "microsoft/phi-2"  # or your chosen model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    return embedder, tokenizer, model


# ================================
# PDF PROCESSING
# ================================
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for file in uploaded_files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ================================
# VECTOR STORE
# ================================
def create_vector_store(chunks, embedder):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings


def retrieve(query, embedder, index, chunks, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)

    return [chunks[i] for i in indices[0]]


# ================================
# GENERATE RESPONSE
# ================================
def generate_response(query, context, tokenizer, model):
    prompt = f"""
You are a research assistant specializing in Deep Eutectic Solvents (DES).

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ================================
# STREAMLIT UI
# ================================
st.set_page_config(page_title="DES Research Chatbot")

st.title("🧪 DES Research Chatbot")

st.write("Upload research papers and ask questions.")

embedder, tokenizer, model = load_models()

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

# File upload
uploaded_files = st.file_uploader(
    "Upload PDF papers",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        text = extract_text_from_pdfs(uploaded_files)
        chunks = chunk_text(text)

        index, embeddings = create_vector_store(chunks, embedder)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("Documents processed!")

# Chat input
query = st.text_input("Ask a question about your papers:")

if query and st.session_state.index is not None:
    with st.spinner("Thinking..."):
        retrieved_chunks = retrieve(
            query,
            embedder,
            st.session_state.index,
            st.session_state.chunks
        )

        context = "\n\n".join(retrieved_chunks)

        response = generate_response(
            query,
            context,
            tokenizer,
            model
        )

    st.subheader("Answer")
    st.write(response)
