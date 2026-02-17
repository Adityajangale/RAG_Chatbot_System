# RAG Chatbot System (LangChain + FAISS + Ollama)

## 📌 Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using semantic search and a local LLM.

The system:
- Loads PDF documents
- Splits text into chunks
- Generates embeddings
- Stores vectors using FAISS
- Retrieves relevant context
- Generates grounded answers using Ollama (phi3:mini)

---

## 🧠 Tech Stack
- Python
- Streamlit
- LangChain
- FAISS (Vector Database)
- Sentence Transformers
- Ollama (phi3:mini model)

---

## ⚙️ Architecture

User Query  
↓  
FAISS Retrieval (Top-k)  
↓  
Prompt Construction  
↓  
LLM (phi3:mini)  
↓  
Final Answer  

---

## 🚀 How to Run Locally

1. Clone the repository:


2. Install dependencies:


3. Install and run Ollama:


4. Run Streamlit:


---

## 📊 Features
- PDF-based knowledge retrieval
- FAISS vector indexing
- Controlled prompt to reduce hallucination
- Latency measurement
- Context-limited retrieval (k=1)
- Token limit for performance optimization

---

## 🔮 Future Improvements
- Hybrid search (BM25 + FAISS)
- Reranking with cross-encoder
- API-based LLM deployment
- Production-ready backend architecture

---

## 👤 Author
Aditya Jangale
