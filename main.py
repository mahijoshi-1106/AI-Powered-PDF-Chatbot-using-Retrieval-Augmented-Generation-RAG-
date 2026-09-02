from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() #  env files loadding
data = PyPDFLoader("document_loaders/fundamentals_of_DEEP_Learning.pdf")
docs = data.load() # data loaded from the text file and stored in the docs variable. The content of the first document is printed, along with the total number of documents loaded.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


template = ChatPromptTemplate.from_messages(
    [
      ("system","you are AI that summarizes the text"),
      ("human","{data}")  
    ]
)



model = ChatMistralAI(
    model = "mistral-small-2506"
)

prompt = template.format_messages(data=chunks)

result = model.invoke(prompt)
print(result.content)