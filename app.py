import streamlit as st
import numpy as np
import faiss
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="DES Research Chatbot", layout="wide")

# ================================
# LOAD MODELS (SAFE FOR CLOUD)
# ================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "google/flan-t5-small"  # lightweight + fast
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.to("cpu")  # force CPU

    return embedder, tokenizer, model


# ================================
# PDF PROCESSING
# ================================
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
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

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index


def retrieve(query, embedder, index, chunks, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)

    return [chunks[i] for i in indices[0]]


# ================================
# GENERATE RESPONSE (FLAN-T5 STYLE)
# ================================
def generate_response(query, context, tokenizer, model):
    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ================================
# UI
# ================================
st.title("🧪 DES Research Chatbot")
st.write("Upload research papers and ask questions.")

embedder, tokenizer, model = load_models()

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================================
# FILE UPLOAD
# ================================
uploaded_files = st.file_uploader(
    "Upload PDF papers",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        try:
            text = extract_text_from_pdfs(uploaded_files)

            if not text.strip():
                st.error("No readable text found in PDFs.")
            else:
                chunks = chunk_text(text)
                index = create_vector_store(chunks, embedder)

                st.session_state.index = index
                st.session_state.chunks = chunks

                st.success("Documents processed!")
        except Exception as e:
            st.error(f"Error processing files: {e}")

# ================================
# CHAT INTERFACE
# ================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about your papers...")

if query:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    if st.session_state.index is None:
        response = "Please upload and process PDFs first."
    else:
        with st.spinner("Thinking..."):
            try:
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

            except Exception as e:
                response = f"Error: {e}"

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)
