from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv() #  env files loadding

docs = [
    Document(
        page_content="""
        Python is a high-level, interpreted programming language known for
        its simple syntax and readability. It is widely used in web development,
        data science, machine learning, automation, and artificial intelligence.
        """,
        metadata={
            "source": "python_notes.pdf",
            "page": 1,
            "topic": "Python"
        }
    ),

    Document(
        page_content="""
        Pandas is a Python library used for data manipulation and analysis.
        It provides data structures such as Series and DataFrame for working
        with structured and tabular data.
        """,
        metadata={
            "source": "pandas_notes.pdf",
            "page": 1,
            "topic": "Pandas"
        }
    ),

    Document(
        page_content="""
        A neural network is a machine learning model consisting of interconnected
        layers of neurons. Neural networks learn patterns from data by adjusting
        weights during training using techniques such as backpropagation and
        gradient descent.
        """,
        metadata={
            "source": "neural_network_notes.pdf",
            "page": 1,
            "topic": "Neural Networks"
        }
    )
]
embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents = docs, 
    embedding = embedding_model, 
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search(query="What is used for data analytics?",k=2)# responsible for retrivering the answer for the question
for r in result:
    print(r.page_content)

retriever = vectorstore.as_retriever()
docs = retriever.invoke("Explain deep learning")
for d in docs:
    print(d.page_content)