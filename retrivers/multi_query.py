from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from dotenv import load_dotenv

load_dotenv()


# 1. Documents
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
    ),
]


# 2. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 3. Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)


# 4. Normal retriever
base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)


# 5. LLM
llm = ChatMistralAI(
    model="mistral-small-2506"
)


# 6. Multi Query Retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)


# 7. Query
query = "What is AI?"


# 8. Retrieve documents
results = retriever.invoke(query)


# 9. Print results
print("\n==============Multi Query Retriever Results ============\n")
for doc in results:
    print(doc.page_content)