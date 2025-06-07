from langchain_core.tools import tool


@tool(response_format="content_and_artifact")
def retrieve(query: str, vector_store) -> tuple[str, list]:
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
