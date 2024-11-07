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
            
            Examples:
            Question: was there any M&As of software companies in 2023?
            Answer:
            Yes, the sources list M&A activity of software companies in 2023.
            - The Society for Plastics Engineers (SPE) acquired 3Dnatives, a digital content, global events, and industry B2B marketing service provider located in Paris, France.
            - Also in 2023, SPE acquired ImplementAM, a provider of learning workshops focused on AM technologies.
            The source also notes that in early March 2023, Nano Dimension made an offer to acquire all outstanding shares of Stratasys. At the time of the source's publication, the fate of the offer had not yet been determined. It is unclear from the source whether Stratasys is a software company.

            Question: did we have any M&A related to machine manufacturers?
            Answer:
            Yes, the sources detail multiple M&A transactions related to machine manufacturers in 2023. 
            M&A of Machine Manufacturers in 2023
            - Nikon Corp. purchased SLM Solutions for $670 million. This acquisition was driven by Nikon's desire to expand its portfolio and grow its business in the aerospace industry, particularly with the U.S. Air Force.
            - Nexa3D acquired Essentium and AddiFab. The acquisition of Essentium allowed Nexa3D to add high-speed extrusion technology to its offerings and expand its reach into military and defense applications. The acquisition of AddiFab further strengthened Nexa3D's position in the market.
            - 3D Systems acquired Wematter for $13.25 million. This acquisition was likely part of 3D Systems' strategy to strengthen its offerings, as noted in the sources.
            - Stratasys acquired Covestro's AM materials business for $46.14 million and was involved in several other significant M&A discussions. This acquisition allowed Stratasys to expand its materials capabilities. Stratasys was also the target of an acquisition attempt by Nano Dimension in early 2023, though the outcome of this offer is not detailed in the sources. The sources also mention that Stratasys was close to two other major deals in 2023, making it a potential driver of consolidation in the industry.
            - Nexa3D acquired XYZprinting's polymer PBF operations. This acquisition allowed Nexa3D to expand its capabilities in polymer powder bed fusion technology.
            The sources also mention that consolidation across machine manufacturers is expected to continue, driven by factors like major players seeking growth and the need for advantages in specific application spaces. Some public companies may attempt to go private due to declining share prices, potentially acquiring assets and aiming to go public again or be acquired by a strategic buyer. Traditional manufacturers may also enter the M&A landscape as supply chains become more important.
            In addition to these specific acquisitions, the sources highlight several companies that are poised to make strategic moves in the future:
            - GE Aerospace, following its corporate breakup, now has more freedom to operate and may be involved in future M&A activity.
            - Formlabs and Markforged, with high confidence from private investors, are also positioned for strategic moves.
            - Nikon SLM Solutions and 3D Systems have options to strengthen their offerings through acquisitions.
            - Nano Dimension and Nexa3D are actively pursuing acquisition strategies.
            The sources suggest that the M&A environment for machine manufacturers and related businesses remains active, with continued consolidation expected.

            
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
