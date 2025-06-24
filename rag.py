# rag.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import docx
import pandas as pd
import fitz  # PyMuPDF

class RAGSearch:
    def __init__(self):
        self.texts = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def add_chunks(self, chunks):
        new_embeddings = self.model.encode(chunks)
        if self.index is None:
            dim = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.embeddings = new_embeddings
            self.texts = chunks
            self.index.add(new_embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.texts.extend(chunks)
            self.index.add(new_embeddings)

    def query(self, query_text, top_k=3):
        if self.index is None:
            return []
        q_emb = self.model.encode([query_text])
        D, I = self.index.search(q_emb, top_k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]


# === TEXT SPLITTER ===
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def load_and_index_file(path, rag: RAGSearch):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_pdf(path)
    elif ext == ".docx":
        text = extract_docx(path)
    elif ext == ".txt":
        text = extract_txt(path)
    elif ext in [".csv", ".xls", ".xlsx"]:
        text = extract_table(path)
    else:
        text = ""
    chunks = splitter.split_text(text)
    rag.add_chunks(chunks)

# === FILE PARSERS ===
def extract_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def extract_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_txt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def extract_table(path):
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return df.to_string(index=False)
    except Exception:
        return ""
