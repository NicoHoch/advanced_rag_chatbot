import getpass
import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.chat_models import init_chat_model
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display


class LanggraphManager:
    def __init__(self, db_connection):
        """Initialize the Langgraph_Manager with OpenAI and PGVector."""
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

        self.db_connection = db_connection  # Store connection params
        self._init_vector_store()
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

    def _init_vector_store(self):
        """(Re)initialize the vector store and embeddings."""

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="my_docs",
            connection=self.db_connection,
        )

    def refresh_connection(self):
        """Refresh the connection to the vector store."""
        self._init_vector_store()

    def index_documents(self):
        """Load, chunk, and index contents of the blog."""
        # Optionally refresh connection before indexing
        self.refresh_connection()
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)
        result = self.vector_store.add_documents(documents=all_splits)
        return result

    def delete_all_documents(self):
        """Delete all documents from the vector store."""
        self.refresh_connection()
        return self.vector_store.delete_collection()

    def delete_documents(self, ids: List[str]):
        """Delete documents from the vector store by their IDs."""
        self.refresh_connection()
        return self.vector_store.delete(ids=ids)

    def create_graph(self):
        memory = MemorySaver()
        vector_store = self.vector_store

        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f" Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        self.agent_executor = create_react_agent(
            self.llm, [retrieve], checkpointer=memory
        )
        with open("graph.png", "wb") as f:
            f.write(self.agent_executor.get_graph().draw_mermaid_png())
        return self.agent_executor
