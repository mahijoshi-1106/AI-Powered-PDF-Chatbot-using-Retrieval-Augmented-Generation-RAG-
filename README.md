# AI-Powered PDF Chatbot

A Streamlit application that lets you upload a PDF and ask questions about its
contents using retrieval-augmented generation (RAG).

## How it works

1. Loads the uploaded PDF with `PyPDFLoader`.
2. Splits the document into overlapping text chunks.
3. Creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Stores the embeddings in ChromaDB.
5. Retrieves the most relevant chunks for each question.
6. Generates an answer with Groq.

## Requirements

- Python 3.14 or newer
- A Groq API key

## Installation

From the project directory, create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirement.txt
```

Alternatively, if you use `uv`:

```powershell
uv sync
```

## Environment setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or expose your API key.

## Run the application

Start Streamlit from the project directory:

```powershell
streamlit run app.py
```

If port `8501` is already in use, specify another port:

```powershell
streamlit run app.py --server.port 8502
```

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Usage

1. Open the Streamlit page.
2. Upload a PDF from the sidebar.
3. Click **Process Book**.
4. Wait for embeddings to be created.
5. Ask questions in the chat box.
6. Expand **View Retrieved Sources** to inspect the document excerpts used.

## ChromaDB notes

The application stores its local vector database in `chroma-db/`. When a new
PDF is processed, the existing collection is cleared and rebuilt.

If another Streamlit process is using the database, stop it with `Ctrl+C` and
restart the application before processing a new PDF.

## Project structure

```text
app.py                 Streamlit user interface and RAG pipeline
create_db.py           Script for creating a ChromaDB from a local PDF
main.py                Command-line RAG example
document_loaders/      Example source documents and loaders
retrivers/             Retriever examples
vector_store/          Vector store utilities
uploads/               Uploaded PDFs
chroma-db/             Local ChromaDB data
```