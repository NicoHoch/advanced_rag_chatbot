import asyncio
import getpass
import os
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import AzureBlobStorageFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from azure.storage.blob import ContainerClient
from openai import AsyncOpenAI
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the vector store for document indexing."""

    def __init__(self, db_connection: str):
        """Initialize VectorStoreManager with database connection and OpenAI setup.

        Args:
            db_connection (str): Database connection string for PGVector.
        """
        self.db_connection = db_connection
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="my_docs",
            connection=self.db_connection,
        )
        self._load_environment()
        logger.info("Vector store initialized successfully")

    def _load_environment(self) -> None:
        """Load and validate environment variables."""
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        if not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING environment variable is not set"
            )

    def refresh_connection(self) -> None:
        """Refresh the connection to the vector store."""
        logger.info("Refreshing vector store connection")
        self._init_vector_store()

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
                # Check if the .txt version exists before attempting to load
                blob_client = container_client.get_blob_client(txt_blob_name)
                if await asyncio.to_thread(blob_client.exists):
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
                else:
                    logger.warning(
                        f"No corresponding .txt file found for {csv_blob_name}"
                    )

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
