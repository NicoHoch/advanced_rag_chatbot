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


class LanggraphManager:
    def __init__(self, db_connection):
        """Initialize the Langgraph_Manager with OpenAI and PGVector."""
        # Ensure environment variables are set
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="my_docs",
            connection=db_connection,
        )

    def index_documents(self):
        """Load, chunk, and index contents of the blog."""
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
        # Index chunks
        result = self.vector_store.add_documents(documents=all_splits)
        return result
