from langchain_community.retrievers import ArxivRetriever


retriever = ArxivRetriever(
    load_max_docs=3,
    load_all_available_meta=True
)


query = "retrieval augmented generation"

docs = retriever.invoke(query)


print(f"Number of documents retrieved: {len(docs)}")


for i, doc in enumerate(docs):
    print("\n" + "=" * 60)
    print(f"DOCUMENT {i + 1}")
    print("=" * 60)

    print("Content:")
    print(doc.page_content[:2000])

    print("\nMetadata:")
    print(doc.metadata)