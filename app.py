import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

CHROMA_DIR = "chroma-db"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📚 Book RAG Assistant")

st.markdown(
    """
    Upload a book/PDF and ask questions about its content.

    **Powered by:**  
    PyPDFLoader → Chunking → HuggingFace Embeddings → ChromaDB → Mistral
    """
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "book_name" not in st.session_state:
    st.session_state.book_name = None


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings()


embedding_model = load_embedding_model()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📖 Upload Your Book")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.info(f"Selected: {uploaded_file.name}")

        if st.button("🚀 Process Book", use_container_width=True):

            # Save uploaded PDF
            pdf_path = os.path.join(
                UPLOAD_DIR,
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress = st.progress(0)

            # --------------------------------------------------
            # STEP 1: LOAD PDF
            # --------------------------------------------------

            st.write("📄 Loading PDF...")

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            progress.progress(25)

            # --------------------------------------------------
            # STEP 2: SPLIT DOCUMENT
            # --------------------------------------------------

            st.write("✂️ Splitting document...")

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            progress.progress(50)

            # --------------------------------------------------
            # STEP 3: REMOVE OLD CHROMA DATABASE
            # --------------------------------------------------

            if os.path.exists(CHROMA_DIR):

                shutil.rmtree(CHROMA_DIR)

            # --------------------------------------------------
            # STEP 4: CREATE CHROMA DATABASE
            # --------------------------------------------------

            st.write("🧠 Creating embeddings...")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=CHROMA_DIR
            )

            progress.progress(75)

            # --------------------------------------------------
            # SAVE VECTORSTORE
            # --------------------------------------------------

            st.session_state.vectorstore = vectorstore
            st.session_state.book_name = uploaded_file.name

            # Clear old chat
            st.session_state.messages = []

            progress.progress(100)

            st.success("✅ Book processed successfully!")

            st.write(f"📄 Pages: {len(docs)}")
            st.write(f"🧩 Chunks: {len(chunks)}")


# --------------------------------------------------
# LOAD EXISTING VECTORSTORE
# --------------------------------------------------

if (
    st.session_state.vectorstore is None
    and os.path.exists(CHROMA_DIR)
):

    try:

        st.session_state.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model
        )

    except Exception:
        st.session_state.vectorstore = None


# --------------------------------------------------
# CHAT SECTION
# --------------------------------------------------

if st.session_state.vectorstore is None:

    st.info(
        "👈 Upload a PDF from the sidebar and click **Process Book** to start."
    )

else:

    st.success(
        f"📚 Current book: **{st.session_state.book_name or 'Uploaded book'}**"
    )

    # --------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    # --------------------------------------------------
    # MISTRAL
    # --------------------------------------------------

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful book assistant.

                Answer the user's question using ONLY the
                information contained in the retrieved context.

                If the answer cannot be found in the context,
                respond with:

                "I don't know based on this book."

                Do not make up information.
                """
            ),
            (
                "human",
                """
                Context:
                {context}

                Question:
                {question}
                """
            )
        ]
    )

    # --------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # --------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------

    user_query = st.chat_input(
        "Ask something about your book..."
    )

    if user_query:

        # Display user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)


        # --------------------------------------------------
        # RETRIEVE DOCUMENTS
        # --------------------------------------------------

        with st.spinner("🔍 Searching the book..."):

            docs = retriever.invoke(user_query)


        # --------------------------------------------------
        # CREATE CONTEXT
        # --------------------------------------------------

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )


        # --------------------------------------------------
        # GENERATE ANSWER
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                formatted_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": user_query
                    }
                )

                response = llm.invoke(
                    formatted_prompt
                )

                answer = response.content

            st.markdown(answer)


        # Save assistant response

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # --------------------------------------------------
        # SHOW SOURCES
        # --------------------------------------------------

        with st.expander("📚 View Retrieved Sources"):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"### Source {i + 1}"
                )

                st.write(
                    doc.page_content[:500]
                )

                if "page" in doc.metadata:

                    st.caption(
                        f"Page: {doc.metadata['page'] + 1}"
                    )