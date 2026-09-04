from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# 1. Load documents
docs = [
    Document(
        page_content="AI enables machines to perform tasks requiring human intelligence.",
        metadata={"source": "doc1"}
    ),
    Document(
        page_content="Machine learning allows computers to learn patterns from data.",
        metadata={"source": "doc2"}
    ),
    Document(
        page_content="Deep learning uses neural networks to learn complex patterns.",
        metadata={"source": "doc3"}
    ),
    Document(
        page_content="RAG combines document retrieval with language models.",
        metadata={"source": "doc4"}
    )
]


# 2. Split documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )

# chunks = text_splitter.split_documents(documents)


# 3. Create embeddings
embeddings = HuggingFaceEmbeddings()


# 4. Store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)


# 5. Create similarity retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        # "fetch_k": 10,
        # "lambda_mult": 0.5
    }
)

print("\n==============Similarity search results ============\n")

# 6. Query
similarity_query = "What is artificial intelligence?"

docs = retriever.invoke(similarity_query)

for doc in docs:
    print(doc.page_content)


print("\n==============MMR search results ============\n")
# 5. Create MMR retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        # "fetch_k": 10,
        # "lambda_mult": 0.5
    }
)

# 6. Query
mmr_query = "What is artificial intelligence?"

docs1 = retriever.invoke(mmr_query)


# 7. Display results
print(f"Number of documents retrieved: {len(docs)}")

for doc in docs1:
    print(doc.page_content)
    