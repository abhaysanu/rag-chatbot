import os
import json
import numpy as np
import easyocr
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# === CONFIG ===
IMAGE_DIR = "images"  # Directory containing image files (png/jpg/jpeg)
reader = easyocr.Reader(['en'], gpu=False)  # Use CPU-only EasyOCR

# === STEP 1: OCR using EasyOCR ===
print("[INFO] Reading text from images using EasyOCR...")
texts = []

for file in sorted(os.listdir(IMAGE_DIR)):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(IMAGE_DIR, file)
        result = reader.readtext(path, detail=0)
        combined_text = "\n".join(result)
        print(f"[INFO] Extracted from {file}:")
        print(combined_text[:200], "...\n")
        texts.append(combined_text)

if not texts:
    raise ValueError("[ERROR] No text extracted from any image. Check image quality or OCR setup.")

# === STEP 2: Split text ===
print("[INFO] Splitting extracted text into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text("\n".join(texts))

documents = [Document(page_content=chunk) for chunk in chunks]

# === STEP 3: Embedding using LangChain wrapper around SentenceTransformer ===
print("[INFO] Embedding chunks using HuggingFaceEmbeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === STEP 4: Create FAISS Vector Store ===
print("[INFO] Storing embeddings in FAISS via LangChain...")
vectorstore = FAISS.from_documents(documents, embedding_model)

# === STEP 5: Save FAISS index and metadata ===
vectorstore.save_local("faiss_index")

with open("chunks.json", "w") as f:
    json.dump({"chunks": chunks}, f)

print("[INFO] Extraction and embedding completed!")


vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
query = "total sales for CLassics in 2008?"
results = vectorstore.similarity_search(query, k=3)  # Returns top 3 most relevant chunks

for i, res in enumerate(results, 1):
    print(f"\n--- Match {i} ---")
    print(res.page_content)