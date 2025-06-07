import asyncio
import getpass
import os
from io import BytesIO
from typing import List, Any, Tuple
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from azure.storage.blob import ContainerClient, BlobClient
from openai import AsyncOpenAI
import logging
from dotenv import load_dotenv

from src.utils.vector_store_manager import VectorStoreManager

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LanggraphAgentManager:
    """Manages the LangGraph agent for data analysis and retrieval."""

    def __init__(self, db_connection: str):
        """Initialize LanggraphAgentManager with database connection and OpenAI setup.

        Args:
            db_connection (str): Database connection string for PGVector.
        """
        self._load_environment()
        self.db_connection = db_connection
        self.llm = init_chat_model(
            "gpt-4o", model_provider="openai", temperature=0
        )  # Changed to gpt-4o for better image generation, temperature 0 for deterministic
        self.async_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.vector_store_manager = VectorStoreManager(db_connection)
        self.agent_executor = None
        self.create_graph()

    def _load_environment(self) -> None:
        """Load and validate environment variables."""
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        if not os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING environment variable is not set"
            )

    async def create_graph(self) -> None:
        """Create a LangGraph agent with retrieval and data analysis tools."""
        memory = MemorySaver()
        vector_store = self.vector_store_manager.vector_store

        @tool(response_format="content_and_artifact")
        def retrieve(query: str) -> Tuple[str, List[Any]]:
            """Retrieve information related to a query. This information include csv files and file locations to be used in the analyze_data tool.

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
        async def analyze_data(query: str, csv_file_name: str) -> List[str]:
            """Analyze data from a CSV file using OpenAI's Code Interpreter.
            Returns text and special markers for generated images.

            Args:
                query (str): The analysis query.
                csv_file_name (str): Name of the CSV file in Azure Blob Storage.

            Returns:
                List[str]: A list of strings, where each string is either
                           text content or a special marker for an image (e.g.,
                           "<IMAGE_GENERATED_FILE_ID>file_id_here</IMAGE_GENERATED_FILE_ID>").
                           The actual image data is retrieved and handled by the FastAPI backend.
            """
            if not csv_file_name.endswith(".csv"):
                return ["Please provide a valid CSV file path."]

            try:
                # Download CSV from Azure Blob Storage
                blob_client: BlobClient = ContainerClient.from_connection_string(
                    conn_str=os.environ["AZURE_STORAGE_CONNECTION_STRING"],
                    container_name="we-are-family",
                ).get_blob_client(csv_file_name)

                # Check if the blob exists before attempting to download
                if not await asyncio.to_thread(blob_client.exists):
                    return [f"File {csv_file_name} not found in Azure Blob Storage."]

                blob_data = await asyncio.to_thread(blob_client.download_blob().readall)
                file_like = BytesIO(blob_data)

                # Upload file to OpenAI
                uploaded_file = await self.async_client.files.create(
                    file=file_like, purpose="assistants"
                )

                # Create assistant with Code Interpreter
                assistant = await self.async_client.beta.assistants.create(
                    instructions=f"You are a data analyst. Analyze the file {csv_file_name} and answer: {query}. If you generate diagrams or images, respond with a textual description of the image and its purpose, then use the Code Interpreter to generate the image.",
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
                    return [f"Run failed with status: {run_status.status}"]

                # Retrieve response messages, including images
                messages_page = await self.async_client.beta.threads.messages.list(
                    thread_id=thread.id, order="asc"
                )
                response_outputs = []
                for message in messages_page.data:
                    if message.role == "assistant":
                        for content_block in message.content:
                            if content_block.type == "text":
                                response_outputs.append(content_block.text.value)
                            elif content_block.type == "image_file":
                                # DO NOT EMBED BASE64 HERE.
                                # Instead, just return the file_id with a special marker.
                                file_id = content_block.image_file.file_id
                                response_outputs.append(
                                    f"This is the file-id to give back to the user for displaying the image. This format needs to be preserved: <IMAGE_GENERATED_FILE_ID>{file_id}</IMAGE_GENERATED_FILE_ID>"
                                )

                # Clean up: Delete the assistant and thread to free up resources
                await self.async_client.beta.assistants.delete(assistant.id)
                await self.async_client.files.delete(
                    uploaded_file.id
                )  # Delete uploaded CSV

                if not response_outputs:
                    return ["No response from assistant."]
                return response_outputs

            except Exception as e:
                logger.error(f"Error in analyze_data tool: {str(e)}")
                return [f"Error processing file: {str(e)}"]

        self.agent_executor = create_react_agent(
            self.llm, [retrieve, analyze_data], checkpointer=memory
        )
        try:
            with open("graph.png", "wb") as f:
                f.write(self.agent_executor.get_graph().draw_mermaid_png())
            logger.info("Agent graph created and saved as graph.png")
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
