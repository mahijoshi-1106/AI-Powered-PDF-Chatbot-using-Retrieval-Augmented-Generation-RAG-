from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # Load environment variables from .env file  

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="chroma-db",  # Directory where the Chroma database is stored
    embedding_function=embedding_model 
) 
print("Number of documents:", vectorstore._collection.count())

retriever = vectorstore.as_retriever(
    search_type="mmr",  # Use similarity search for retrieval
    search_kwargs={
        "k": 4,
        "fetch_k":10 , # Retrieve the top 4 most similar documents
        "lambda_mult": 0.5  # Balance between relevance and diversity in retrieval
    }
)

llm = ChatMistralAI(
    model="mistral-small-2506" # Specify the Mistral model to use
)

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",""" You are a helpful assistant that provides information based on the retrieved documents. 
        If the answer is not found in the documents, respond with "I don't know." """),
        ("human", """context: {context}
        Question: {question}""")
    ]
)

print("Rag system created")

print("press 0 to exit")
while True:
    query = input("YOU: ")
    if query == "0":
        break
    docs = retriever.invoke(query)

    print("\n--- RETRIEVED DOCUMENTS ---")

    for i, doc in enumerate(docs):
        print(f"\n===== DOCUMENT {i + 1} =====")
        print(doc.page_content[:100])

    print("\n--- END DOCUMENTS ---")

    context = "\n".join([doc.page_content for doc in docs])
    response = prompt.invoke({
        "context": context,
        "question": query
    })
    
    final_response = llm.invoke(response)
    print(f"\nAI: {final_response.content}\n")