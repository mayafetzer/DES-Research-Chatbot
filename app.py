import streamlit as st
import numpy as np
import faiss
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="DES Research Chatbot", layout="wide")

# ================================
# LOAD MODELS
# ================================
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return embedder, tokenizer, model


# ================================
# PDF CLEANING
# ================================
def clean_text(text):
    # Remove references section (common issue)
    text = re.split(r"(?i)references", text)[0]

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove weird artifacts
    text = text.replace("\n", " ")

    return text.strip()


def extract_text_from_pdfs(files):
    text = ""

    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    return clean_text(text)


# ================================
# SMART CHUNKING
# ================================
def chunk_text(text, chunk_size=400, overlap=100):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ================================
# VECTOR STORE
# ================================
def create_vector_store(chunks, embedder):
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, embeddings


def retrieve(query, embedder, index, chunks, k=5):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)

    retrieved = [chunks[i] for i in indices[0]]

    # Filter out very short/noisy chunks
    retrieved = [c for c in retrieved if len(c) > 50]

    return retrieved


# ================================
# GENERATE RESPONSE
# ================================
def generate_response(query, context, tokenizer, model):
    prompt = f"""
You are a scientific research assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3  # lower = more factual
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
        text = extract_text_from_pdfs(uploaded_files)

        if not text:
            st.error("Could not extract text from PDFs.")
        else:
            chunks = chunk_text(text)
            index, embeddings = create_vector_store(chunks, embedder)

            st.session_state.index = index
            st.session_state.chunks = chunks

            st.success(f"Processed {len(chunks)} chunks.")

# ================================
# CHAT UI
# ================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask a question about your papers...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    if st.session_state.index is None:
        response = "Please upload PDFs first."
    else:
        with st.spinner("Thinking..."):
            retrieved_chunks = retrieve(
                query,
                embedder,
                st.session_state.index,
                st.session_state.chunks
            )

            context = "\n\n".join(retrieved_chunks)

            # 🔍 DEBUG VIEW (VERY IMPORTANT)
            with st.expander("🔍 Retrieved Context"):
                st.write(retrieved_chunks)

            response = generate_response(
                query,
                context,
                tokenizer,
                model
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.write(response)
