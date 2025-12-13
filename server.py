import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import tkinter as tk
from tkinter import scrolledtext

# -----------------------
# 1. Load PDFs and chunk text
# -----------------------
def pdf_to_chunks(pdf_path, chunk_size=500):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    # Split text into chunks
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Ingest all PDFs in folder
all_chunks = []
pdf_folder = r"D:\AETMA_prjs\AI"
for fname in os.listdir(pdf_folder):
    if fname.endswith(".pdf"):
        path = os.path.join(pdf_folder, fname)
        chunks = pdf_to_chunks(path)
        all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

# -----------------------
# 2. Embed chunks
# -----------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)

# -----------------------
# 3. Build FAISS index
# -----------------------
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(chunk_embeddings)
print(f"FAISS index contains {index.ntotal} vectors")

# -----------------------
# 4. Query functions
# -----------------------
def query_llama3(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def answer_question(question, top_k=5):
    q_emb = embedding_model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    retrieved_chunks = [all_chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved_chunks)
    system_prompt = """
You are a reviewer and analyzer. Make good comparison and do not be very polite,
you must pick the best applicant for a job.
Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."
"""
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}"
    return query_llama3(prompt)

# -----------------------
# 5. GUI with Tkinter
# -----------------------
def on_ask():
    question = entry.get("1.0", tk.END).strip()
    if question:
        response_box.delete("1.0", tk.END)
        response_box.insert(tk.END, "Thinking...\n")
        root.update()
        answer = answer_question(question)
        response_box.delete("1.0", tk.END)
        response_box.insert(tk.END, answer)

root = tk.Tk()
root.title("AI PDF Assistant")

# Input
tk.Label(root, text="Your Question:").pack(anchor="w")
entry = tk.Text(root, height=3, width=80)
entry.pack(padx=5, pady=5)

# Button
ask_button = tk.Button(root, text="Ask AI", command=on_ask)
ask_button.pack(pady=5)

# Response
tk.Label(root, text="AI Answer:").pack(anchor="w")
response_box = scrolledtext.ScrolledText(root, height=15, width=80, wrap=tk.WORD)
response_box.pack(padx=5, pady=5)

root.mainloop()
