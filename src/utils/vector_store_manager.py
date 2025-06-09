import asyncio
import getpass
import os
import base64
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_community.document_loaders import AzureBlobStorageFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from azure.storage.blob import ContainerClient, BlobClient
from langchain_community.document_loaders import AzureBlobStorageFileLoader
from langchain.schema import Document
from pdf2image import convert_from_bytes
import tempfile
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field

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

            client = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")

            docs = []

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

            # Load PDFs plain text
            for blob_name in pdf_blobs:
                loader = AzureBlobStorageFileLoader(
                    conn_str=azure_connection_string,
                    container=container_name,
                    blob_name=blob_name,
                )
                docs.extend(await asyncio.to_thread(loader.load))

                # PDF als Bytes laden
                blob_client = BlobClient.from_connection_string(
                    azure_connection_string, container_name, blob_name
                )
                pdf_bytes = blob_client.download_blob().readall()

                # Konvertiere Seiten zu Bildern
                pages = convert_from_bytes(pdf_bytes, dpi=150)
                context_buffer = ""

                for page_num, image in enumerate(pages):
                    try:
                        # Temporäre Bilddatei
                        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                            image.save(tmp.name, format="PNG")
                            with open(tmp.name, "rb") as img_file:
                                image_b64 = base64.b64encode(img_file.read()).decode(
                                    "utf-8"
                                )

                        # Define schema as a class
                        class ImageDescriptionSchema(BaseModel):
                            """Schema for describing the image content of a PDF page."""

                            tags: List[str] = Field(
                                description="3-5 relevant keywords describing the image content."
                            )
                            context_needed: bool = Field(
                                description="True if context from the previous page is needed to understand the image content, False otherwise."
                            )
                            page: int = Field(
                                description="The page number of the current page in the PDF document."
                            )
                            description: str = Field(
                                description="A detailed description of the image content, including visual elements like diagrams, charts, or graphics."
                            )
                            context: str = Field(
                                description="Context information required for understanding the next page."
                            )

                        # Bind schema to model
                        model_with_structure = client.with_structured_output(
                            ImageDescriptionSchema
                        )
                        # Vision Prompt
                        vision_prompt = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"Act as my image-to-text translator. "
                                            f"Please provide a comprehensive description of this page's contents. "
                                            "Pay special attention to and detail the following elements:\n"
                                            "1. Diagrams, charts, graphs, and other visual elements that text extraction cannot capture\n"
                                            "2. Detailed number of the charts and what you read out of them\n"
                                            "3. Key figures, statistics, or numerical data shown in visualizations\n"
                                            "4. Color schemes and their potential significance\n"
                                            "5. Headers, subheaders, and their hierarchical relationship\n"
                                            "6. Any tables or structured data present"
                                            + (
                                                f"\nContext from previous page: {context_buffer}"
                                                if context_buffer
                                                else ""
                                            )
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_b64}"
                                        },
                                    },
                                ],
                            }
                        ]
                        # Use langgraph's invoke method for structured output
                        vision_response = model_with_structure.invoke(vision_prompt)

                        description = vision_response.description
                        tags = vision_response.tags
                        context_needed = vision_response.context_needed
                        extracted_page = vision_response.page

                        full_description = (
                            f"Page {extracted_page}: {description}\n"
                            f"Tags: {', '.join(tags)}\n"
                        )

                        # Add context information if needed
                        if context_needed and context_buffer:
                            full_description = f"Context from previous page: {context_buffer}\n{full_description}"

                        # Update context buffer for next iteration
                        context_buffer = vision_response.context

                        vision_doc = Document(
                            page_content=full_description,
                            metadata={
                                "type": "vision_description",
                                "file_page": page_num + 1,
                                "extracted_page": extracted_page,
                                "source": blob_name,
                                "has_context": context_needed,
                            },
                        )
                        docs.append(vision_doc)

                    except Exception as e:
                        logger.error(
                            f"Error processing page {page_num + 1} of {blob_name}: {str(e)}"
                        )

            csv_blobs = [
                blob.name
                for blob in container_client.list_blobs()
                if blob.name.lower().endswith(".csv")
            ]

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
