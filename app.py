from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np, pickle, os

# ✅ FastAPI app define karna zaroori hai
app = FastAPI()

# ✅ CORS (taake frontend connect kar sake)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Model load
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ✅ Data load
embeddings, documents = None, []
try:
    embeddings = np.load("data/index/embeddings.npy")
    with open("data/index/docs.pkl", "rb") as f:
        documents = pickle.load(f)
    print(f"✅ Loaded {len(documents)} documents into memory")
except Exception as e:
    print(f"⚠️ Could not load index: {e}")

# ✅ API request schema
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "✅ Ustad Amil AI backend chal raha hai!"}

@app.post("/ask")
def ask_question(req: QueryRequest):
    if embeddings is None or not documents:
        return {"answer": "⚠️ Index missing. Pehle ingest_all.py run karo.", "source": "N/A"}

    q_embed = model.encode(req.question)
    sims = np.dot(embeddings, q_embed) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embed))
    top_idx = int(np.argmax(sims))

    best = documents[top_idx]
    return {"answer": best["text"], "source": best["source"]}
