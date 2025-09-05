import os, pickle
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

# Paths
pdf_dir = "data/pdfs"
index_dir = "data/index"
os.makedirs(index_dir, exist_ok=True)

# Model
print("📥 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

documents = []
embeddings = []

# Read PDFs
print(f"📖 Ingesting PDFs from: {pdf_dir}")
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_dir, file)
        print(f"➡️ Reading: {file}")
        try:
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]
            
            for c in chunks:
                documents.append({"text": c, "source": file})
                embeddings.append(model.encode(c, convert_to_numpy=True))
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

if not documents:
    print("❌ No PDF content found! Please put PDFs inside data/pdfs/")
    exit()

embeddings = np.array(embeddings, dtype="float32")

# Save outputs
docs_path = os.path.join(index_dir, "docs.pkl")
emb_path = os.path.join(index_dir, "embeddings.npy")

with open(docs_path, "wb") as f:
    pickle.dump(documents, f)
np.save(emb_path, embeddings)

print(f"\n✅ Indexing complete!")
print(f"   → Saved documents: {docs_path}")
print(f"   → Saved embeddings: {emb_path}")
print(f"   → Total chunks: {len(documents)}")
print("   → Example Source:", documents[0]["source"])
