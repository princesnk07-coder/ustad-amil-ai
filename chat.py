import streamlit as st
import pickle, os
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------
# Load model & data
# -----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@st.cache_resource
def load_data():
    try:
        embeddings = np.load("data/index/embeddings.npy")
        with open("data/index/docs.pkl", "rb") as f:
            documents = pickle.load(f)
        return embeddings, documents
    except Exception as e:
        st.error(f"⚠️ Could not load data: {e}")
        return None, None

model = load_model()
embeddings, documents = load_data()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="📖 Ustad Amil AI", layout="wide")
st.title("🕌 Ustad Amil AI (Demo)")
st.write("Ask questions and get answers from sample Islamic texts (demo PDF included).")

question = st.text_input("❓ Sawal likhiye:")

if st.button("🔍 Pucho"):
    if not question.strip():
        st.warning("Pehle apna sawal likhiye.")
    elif embeddings is None or not documents:
        st.error("⚠️ Data not found! Run ingest_all.py first or add PDFs in data/pdfs/")
    else:
        try:
            q_embed = model.encode(question)
            sims = np.dot(embeddings, q_embed) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embed)
            )
            top_idx = int(np.argmax(sims))
            best = documents[top_idx]

            st.markdown("### ✅ Jawab:")
            st.write(best["text"])
            st.markdown(f"📖 **Source:** {best['source']}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
