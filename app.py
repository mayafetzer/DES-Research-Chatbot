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

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 PDF Research Chatbot (RAG + LLM)")

# ================================
# CACHE MODELS
# ================================
@st.cache_resource
def load_embedding_model():
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


embedding_model = load_embedding_model()
tokenizer, model = load_llm()


# ================================
# PDF PROCESSING
# ================================
def extract_text_from_pdfs(pdf_files):
    text = ""
    for file in pdf_files:
        reader = PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


# ================================
# BUILD VECTOR DB
# ================================
def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks


# ================================
# RETRIEVAL
# ================================
def retrieve(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]


# ================================
# CHAT FUNCTION
# ================================
def generate_answer(query, index, chunks):

    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a scientific research assistant.

Use ONLY the provided context from the paper to answer.
If not found, say: "The paper does not provide this information."

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response[len(prompt):].strip()


# ================================
# SIDEBAR - UPLOAD PDFs
# ================================
st.sidebar.header("📂 Upload PDFs")
pdf_files = st.sidebar.file_uploader(
    "Upload research papers",
    type=["pdf"],
    accept_multiple_files=True
)

index = None
chunks = None

if pdf_files:
    with st.spinner("Reading PDFs..."):
        text = extract_text_from_pdfs(pdf_files)

    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)

    with st.spinner("Building vector index..."):
        index, chunks = build_faiss_index(chunks)

    st.success("Knowledge base ready!")

# ================================
# CHAT UI
# ================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a question about your papers:")

if user_query:

    if index is None:
        st.warning("Please upload PDFs first.")
    else:
        with st.spinner("Thinking..."):
            answer = generate_answer(user_query, index, chunks)

        st.session_state.chat_history.append((user_query, answer))


# ================================
# DISPLAY CHAT HISTORY
# ================================
for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**🤖 Bot:** {a}")
    st.markdown("---")
