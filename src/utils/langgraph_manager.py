import asyncio
import getpass
import os
from io import BytesIO
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import AzureBlobStorageFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from azure.storage.blob import ContainerClient
import openai
from openai import AsyncOpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LanggraphManager:
    """Manages document indexing and agent-based data analysis using LangChain and OpenAI Assistants API."""

    def __init__(self, db_connection: str):
        """Initialize LanggraphManager with database connection and OpenAI setup.

        Args:
            db_connection (str): Database connection string for PGVector.
        """
        self._load_environment()
        self.db_connection = db_connection
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._init_vector_store()
        self.agent_executor = None

    def _load_environment(self) -> None:
        """Load and validate environment variables."""
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        if not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING environment variable is not set"
            )

    def _init_vector_store(self) -> None:
        """Initialize the vector store and embeddings."""
        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            self.vector_store = PGVector(
                embeddings=self.embeddings,
                collection_name="my_docs",
                connection=self.db_connection,
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def refresh_connection(self) -> None:
        """Refresh the connection to the vector store."""
        logger.info("Refreshing vector store connection")
        self._init_vector_store()

    async def index_documents(self) -> Optional[List[str]]:
        """Load, chunk, and index documents from Azure Blob Storage.

        Returns:
            Optional[List[str]]: List of document IDs indexed, or None if indexing fails.
        """
        try:
            self.refresh_connection()
            azure_connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
            container_name = "we-are-family"

            container_client = ContainerClient.from_connection_string(
                conn_str=azure_connection_string,
                container_name=container_name,
            )

            # List blobs for PDFs and CSVs
            pdf_blobs = [
                blob.name
                for blob in container_client.list_blobs()
                if blob.name.lower().endswith(".pdf")
            ]
            csv_blobs = [
                blob.name
                for blob in container_client.list_blobs()
                if blob.name.lower().endswith(".csv")
            ]
            docs = []

            # Load PDFs
            for blob_name in pdf_blobs:
                loader = AzureBlobStorageFileLoader(
                    conn_str=azure_connection_string,
                    container=container_name,
                    blob_name=blob_name,
                )
                docs.extend(await asyncio.to_thread(loader.load))

            # Load CSVs as text
            for csv_blob_name in csv_blobs:
                txt_blob_name = csv_blob_name.replace(".csv", ".txt")
                loader = AzureBlobStorageFileLoader(
                    conn_str=azure_connection_string,
                    container=container_name,
                    blob_name=txt_blob_name,
                )
                csv_docs = await asyncio.to_thread(loader.load)
                for doc in csv_docs:
                    doc.metadata["filetype"] = "csv"
                    doc.metadata["csv_file_name"] = csv_blob_name
                docs.extend(csv_docs)

            # Split and index documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            all_splits = text_splitter.split_documents(docs)
            result = await asyncio.to_thread(
                self.vector_store.add_documents, all_splits
            )
            logger.info(f"Indexed {len(result)} document chunks")
            return result

        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return None

    async def delete_all_documents(self) -> bool:
        """Delete all documents from the vector store.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        try:
            self.refresh_connection()
            await asyncio.to_thread(self.vector_store.delete_collection)
            logger.info("All documents deleted from vector store")
            return True
        except Exception as e:
            logger.error(f"Error deleting all documents: {str(e)}")
            return False

    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from the vector store by their IDs.

        Args:
            ids (List[str]): List of document IDs to delete.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        try:
            self.refresh_connection()
            await asyncio.to_thread(self.vector_store.delete, ids=ids)
            logger.info(f"Deleted documents with IDs: {ids}")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False

    async def create_graph(self) -> None:
        """Create a LangGraph agent with retrieval and data analysis tools."""
        memory = MemorySaver()
        vector_store = self.vector_store

        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query.

            Args:
                query (str): The search query.

            Returns:
                tuple: Serialized content and retrieved documents.
            """
            try:
                retrieved_docs = vector_store.similarity_search(query, k=2)
                serialized = "\n\n".join(
                    f"Source: {doc.metadata}\nContent: {doc.page_content}"
                    for doc in retrieved_docs
                )
                return serialized, retrieved_docs
            except Exception as e:
                logger.error(f"Error in retrieve tool: {str(e)}")
                return f"Error retrieving documents: {str(e)}", []

        @tool(response_format="content")
        async def analyze_data(query: str, csv_file_name: str):
            """Analyze data from a CSV file using OpenAI's Code Interpreter.

            Args:
                query (str): The analysis query.
                csv_file_name (str): Name of the CSV file in Azure Blob Storage.

            Returns:
                str: Analysis result or error message.
            """
            if not csv_file_name.endswith(".csv"):
                return "Please provide a valid CSV file path.", None

            try:
                # Download CSV from Azure Blob Storage
                blob_client = ContainerClient.from_connection_string(
                    conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
                    container_name="we-are-family",
                ).get_blob_client(csv_file_name)
                blob_data = await asyncio.to_thread(blob_client.download_blob().readall)
                file_like = BytesIO(blob_data)

                # Upload file to OpenAI
                uploaded_file = await self.async_client.files.create(
                    file=file_like, purpose="assistants"
                )

                # Create assistant with Code Interpreter
                assistant = await self.async_client.beta.assistants.create(
                    instructions=f"You are a data analyst. Analyze the file {csv_file_name} and answer: {query}",
                    model="gpt-4o",
                    tools=[{"type": "code_interpreter"}],
                    tool_resources={
                        "code_interpreter": {"file_ids": [uploaded_file.id]}
                    },
                )

                # Create and run thread
                thread = await self.async_client.beta.threads.create()
                run = await self.async_client.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                    instructions=query,
                )

                # Poll for run completion
                while True:
                    run_status = await self.async_client.beta.threads.runs.retrieve(
                        thread_id=thread.id, run_id=run.id
                    )
                    if run_status.status in ["completed", "failed", "cancelled"]:
                        break
                    await asyncio.sleep(2)

                if run_status.status != "completed":
                    return f"Run failed with status: {run_status.status}", None

                # Retrieve response
                messages = await self.async_client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                if messages.data and messages.data[0].content:
                    answer = messages.data[0].content[0].text.value
                    return answer
                return "No response from assistant.", None

            except Exception as e:
                logger.error(f"Error in analyze_data tool: {str(e)}")
                return f"Error processing file: {str(e)}", None

        self.agent_executor = create_react_agent(
            self.llm, [retrieve, analyze_data], checkpointer=memory
        )
        try:
            with open("graph.png", "wb") as f:
                f.write(self.agent_executor.get_graph().draw_mermaid_png())
            logger.info("Agent graph created and saved as graph.png")
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
