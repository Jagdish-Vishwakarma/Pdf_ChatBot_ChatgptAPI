import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time

# Set OpenAI API key from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to extract all text from pre-uploaded PDF
def get_pre_uploaded_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def generate_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(text)

# Convert chunks into vectors using OpenAI Embeddings (FAISS index in-memory) with batching
def chunks_to_vectors(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = None
    batch_size = 5
    for i in range(0, len(chunks), batch_size):
        chunk_batch = chunks[i:i+batch_size]
        if vector_store is None:
            vector_store = FAISS.from_texts(chunk_batch, embeddings)
        else:
            batch_vector_store = FAISS.from_texts(chunk_batch, embeddings)
            vector_store.merge_from(batch_vector_store)
        time.sleep(60)
    return vector_store

# Main app
def app():
    st.title("ASTM Documents Chatbot")
    st.sidebar.info("This chatbot answers questions based on a pre-configured ASTM document. Enter your query below.")
    
    pre_uploaded_pdf_path = "Investment_and_MA.pdf"
    
    if 'vector_store' not in st.session_state:
        with st.spinner("Configuring pre-uploaded document... ⏳"):
            raw_text = get_pre_uploaded_pdf_text(pre_uploaded_pdf_path)
            if raw_text is None or raw_text.strip() == "":
                st.error("No text extracted from the pre-uploaded PDF. Please check the file.")
                return

            chunks = generate_chunks(raw_text)
            vector_store = chunks_to_vectors(chunks)
            st.session_state.vector_store = vector_store
            st.success("Pre-uploaded document processed. You can now ask questions.")

    user_question = st.text_input("Ask a question based on the pre-uploaded document")
    if user_question and 'vector_store' in st.session_state:
        docs = st.session_state.vector_store.similarity_search(user_question)
        context = " ".join([doc.page_content for doc in docs])
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_question}"}
            ],
            temperature=0.3
        )
        st.write("Reply: ", response['choices'][0]['message']['content'])

if __name__ == "__main__":
    app()
