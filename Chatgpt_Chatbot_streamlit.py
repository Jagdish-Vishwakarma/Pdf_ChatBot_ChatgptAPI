import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Fetch the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

# Main app
def app():
    # Display a centered logo at the top
    st.image("logo.svg", use_column_width=True, caption="")

    # Check if the API key is loaded and display a message
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set it in Streamlit secrets.")
    else:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        st.write("API key loaded successfully.")

    st.title("ASTM ChatBot")
    st.sidebar.title("ASTM Application Information")
    
    # Description about the data in the sidebar
    st.sidebar.write("""
    This application provides answers based on pre-uploaded ASTM documents related to Corporate Investments and Mergers and Acquisitions.
    
    It uses AI to retrieve context-specific responses from a pre-configured document.
    """)

    # Specify the path to the pre-uploaded PDF
    pre_uploaded_pdf_path = "Investment and M&A.pdf"  # Update path or filename if needed

    # This will store the FAISS index globally for the session
    if 'vector_store' not in st.session_state:
        with st.spinner("Configuring pre-uploaded document... ⏳"):
            # Get the extracted text from the pre-uploaded PDF
            raw_text = get_pdf_text(pre_uploaded_pdf_path)

            # Check if the extracted text is None or empty
            if raw_text is None or raw_text.strip() == "":
                st.error("No text extracted from the PDF. Please check the file.")
                return

            # Split text into chunks and create FAISS index
            chunks = generate_chunks(raw_text)
            vector_store = chunks_to_vectors(chunks)
            st.session_state.vector_store = vector_store  # Store FAISS index in session state
            st.success("Document processed. You can now ask questions.")

    # User Input 
    user_question = st.text_input("Ask a question based on the document")

    if user_question:
        if 'vector_store' in st.session_state:
            # Use the vector store stored in the session state
            user_input(user_question, st.session_state.vector_store)
        else:
            st.warning("Please wait while the document is being processed.")

# Extract all text from a pre-uploaded PDF
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    
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

# Convert chunks into vectors using OpenAI Embeddings (FAISS index in-memory)
def chunks_to_vectors(chunks):
    embeddings = OpenAIEmbeddings()
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

if __name__ == "__main__":
    app()
