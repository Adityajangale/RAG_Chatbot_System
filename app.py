import streamlit as st
from rag_pipeline import create_rag_chain
import os
import time

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot (phi-3)")

# -------------------------------
# PDF Path
# -------------------------------
PDF_PATH = os.path.join("Data", "Dataset.pdf")

# -------------------------------
# Load RAG (cached)
# -------------------------------
@st.cache_resource
def load_rag():
    return create_rag_chain(PDF_PATH)

qa_chain = load_rag()

# -------------------------------
# Session State for Chat Memory
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# User Input
# -------------------------------
question = st.text_input("Ask me Anything")

if question:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.spinner("Thinking..."):
        start_time = time.time()

        response = qa_chain.invoke({"query": question})
        answer = response["result"]

        end_time = time.time()
        latency = round(end_time - start_time, 2)

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Show latency
    st.caption(f"⏱️ Response time: {latency} seconds")

# -------------------------------
# Display Conversation (Last 10)
# -------------------------------
st.subheader("Conversation")

for msg in st.session_state.messages[-10:]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")
