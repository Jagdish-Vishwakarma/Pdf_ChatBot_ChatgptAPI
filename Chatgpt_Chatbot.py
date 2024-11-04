import os
import pickle
import time
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI  # Import ChatOpenAI for chat-based models

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
else:
    os.environ['OPENAI_API_KEY'] = openai_api_key
    st.write("API key loaded successfully.")

# Extract all text from pre-uploaded PDF
def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)

        # Iterate through all the pages in the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

# Split text into chunks
def generate_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200  # Reduced overlap for efficiency
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Convert chunks into vectors using OpenAI Embeddings (FAISS index in-memory) with batching
def chunks_to_vectors(chunks, batch_size=5):
    # Initialize the embeddings model
    embeddings = OpenAIEmbeddings()

    # Create FAISS index from the text chunks
    vector_store = FAISS.from_texts(chunks, embeddings)
   
    return vector_store


# Get conversation chain
def get_conversation(vector_store):
    prompt_template = """
    All the files uploaded are directly or indirectly related to Additive Manufacturing and 3D Printing industry and companies.
    Answer the question that is asked with as much detail as you can, given the context that has been provided. 
    If you are unable to provide an answer based on the provided context, simply say 
    'Answer cannot be provided based on the context that has been provided', instead of forcing an answer.
    
    Context: \n{context}\n
    Question: \n{question}\n
    Answer:
    """
    
    # Initialize the OpenAI LLM using the ChatOpenAI class for chat-based models
    model = ChatOpenAI(model="gpt-4", temperature=0.9, max_tokens=1000)
    
    # Create the prompt
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Set up the retriever using the FAISS vector store
    retriever = vector_store.as_retriever()

    # Return the RetrievalQA chain
    return RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", return_source_documents=True)




# Handle user input
def user_input(question, vector_store):
    # Get the conversation chain
    chain = get_conversation(vector_store)

    # Perform the retrieval and get the response
    response = chain({"query": question})

    st.write("Reply: ", response["result"])

# Main app
def app():
    st.title("ASTM Documents Chatbot")
    st.sidebar.title("Upload Documents")

    # Sidebar: Upload PDF documents
    pdf_docs = st.sidebar.file_uploader("Upload your documents in PDF format, then click on Chat Now.", accept_multiple_files=True)

    analyze_triggered = st.sidebar.button("Chat Now")

    # This will store the FAISS index globally for the session
    if analyze_triggered:
        with st.spinner("Configuring... ⏳"):
            # Get the extracted text from the PDFs
            raw_text = get_pdf_text(pdf_docs)

            # Check if the extracted text is None or empty
            if raw_text is None or raw_text.strip() == "":
                st.error("No text extracted from the PDF. Please check the file.")
                return

            # Split text into chunks and create FAISS index
            chunks = generate_chunks(raw_text)
            vector_store = chunks_to_vectors(chunks)
            st.session_state.vector_store = vector_store  # Store FAISS index in session state
            st.success("Documents processed. You can now ask questions.")

    # User Input 
    user_question = st.text_input("Ask a question based on the documents that were uploaded")

    if user_question:
        if 'vector_store' in st.session_state:
            # Use the vector store stored in the session state
            user_input(user_question, st.session_state.vector_store)
        else:
            st.warning("Please upload and process the documents first.")

if __name__ == "__main__":
    app()
