import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load password and API key from Streamlit secrets
PASSWORD = st.secrets.get("APP_PASSWORD")
openai_api_key = st.secrets.get("OPENAI_API_KEY")

# Check if both secrets are set
if PASSWORD is None:
    st.error("APP_PASSWORD is not set in the Streamlit secrets.")
elif openai_api_key is None:
    st.error("OPENAI_API_KEY is not set in the Streamlit secrets.")
else:
    # Password authentication
    def authenticate():
        st.sidebar.header("Login")
        password = st.sidebar.text_input("Enter password", type="password")
        if password == PASSWORD:
            return True
        else:
            st.sidebar.error("Incorrect password")
            return False

    # Only run the main app if the user is authenticated
    if authenticate():
        st.title("Welcome to the secure app!")

        # Main app function
        def app():
            # Set up the OpenAI API key
            os.environ['OPENAI_API_KEY'] = openai_api_key

            # Display a centered logo at the top
            st.image("logo.svg", use_column_width=True, caption="")

            st.title("ASTM ChatBot")
            st.sidebar.title("ASTM Application Information")
            
            # Description about the data in the sidebar
            st.sidebar.write("""
            This application provides answers based on pre-uploaded ASTM documents related to Corporate Investments, Mergers, and Acquisitions(M&A).
            
            It uses AI to retrieve context-specific responses from a pre-configured document.
            """)

            # Specify the path to the pre-uploaded PDF
            pre_uploaded_pdf_path = "Investment and M&A.pdf"

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
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        # Split text into chunks
        def generate_chunks(text):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
            return text_splitter.split_text(text)

        # Convert chunks into vectors
        def chunks_to_vectors(chunks):
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embeddings)
            return vector_store

        # Get conversation chain
        def get_conversation(vector_store):
            prompt_template = """
            The files uploaded are related to Corporate Investments and Mergers and Acquisitions.
            
            Answer the question with as much detail as possible, providing in-depth context. If the answer is complex, break it into multiple paragraphs or bullet points.
            If you cannot find an answer based on the context, reply: "Answer cannot be provided based on the context provided."
            
            Context: \n{context}\n
            Question: \n{question}\n
            Answer:
            """
            model = ChatOpenAI(model="gpt-4", temperature=0.6, max_tokens=1500, frequency_penalty=0.2)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            return RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", return_source_documents=True)

        # Handle user input
        def user_input(question, vector_store):
            chain = get_conversation(vector_store)
            response = chain({"query": question})
            st.write("Reply: ", response["result"])

        # Run the app
        app()
