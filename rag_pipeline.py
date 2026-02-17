import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

INDEX_PATH = "faiss_index"

def create_rag_chain(pdf_path):
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load or build FAISS index (ONE TIME ONLY)
    if os.path.exists(INDEX_PATH):
        vectordb = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(docs)

        vectordb = FAISS.from_documents(chunks, embeddings)
        vectordb.save_local(INDEX_PATH)

    # LLM
    llm = Ollama(
        model="phi3:mini",
        temperature=0,
        num_predict=150
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.

- If the question refers to summarizing the conversation, summarize ONLY the topic explicitly mentioned.
- If the question is related to the dataset, answer using ONLY the dataset.
- If the question is NOT related to the dataset, clearly say:
  "This question is not related to the provided dataset, but here is a general answer:"

Context:
{context}

Question:
{question}

Answer:
"""
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain
