import asyncio
import getpass
import os
from io import BytesIO
from typing import Annotated, List, Optional, Dict, Any
import uuid
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from langchain_core.messages import AIMessage, BaseMessage
import json
import re
import base64
import openai
from src.models.login import LoginRequest
from src.utils.postgres_manager import PostgresManager
from src.utils.langgraph_manager import LanggraphAgentManager
from src.utils.vector_store_manager import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBasic()

# Initialize PostgresManager and LanggraphAgentManager
db_manager = PostgresManager()
vector_store_manager = VectorStoreManager(db_manager.db_uri)
langgraph_manager = LanggraphAgentManager(db_manager.db_uri)
graph = None  # Global graph variable to store agent_executor


@app.on_event("startup")
async def startup_event():
    """Initialize database and LangGraph agent on startup."""
    try:
        logger.info("Initializing database and LangGraph agent")
        db_manager.create_tables()  # Create database tables
        # await langgraph_manager.index_documents() # Commented out for faster startup
        await langgraph_manager.create_graph()  # Create the LangGraph agent
        global graph
        graph = langgraph_manager.agent_executor  # Assign agent_executor to graph
        if graph is None:
            raise ValueError(
                "Failed to initialize LangGraph agent: agent_executor is None"
            )
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    try:
        logger.info("Closing database connection")
        db_manager.close()
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> str:
    """Authenticate user against the PostgreSQL database.

    Args:
        credentials: HTTP Basic credentials containing username and password.

    Returns:
        str: Authenticated username.

    Raises:
        HTTPException: If authentication fails.
    """
    username = credentials.username
    password = credentials.password
    if not db_manager.verify_user(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return username


class MessageRequest(BaseModel):
    """Pydantic model for chat request payload."""

    message: str
    session_id: str


@app.post("/session_id")
def get_session_id(username: Annotated[str, Depends(get_current_username)]):
    """Get the session ID for the user.

    Args:
        username: The authenticated username.

    Returns:
        str: The session ID.
    """

    session_id = generate_session_id(username)
    return {"session_id": session_id}


def generate_session_id(username: str) -> str:
    """Generate a unique session ID for the chat.

    Returns:
        str: A unique session ID.
    """
    return f"chat_{username}_{uuid.uuid4()}"


@app.post("/chat")
async def chat(
    request: MessageRequest, username: Annotated[str, Depends(get_current_username)]
):
    """Stream chat responses from the LangGraph agent, including text and images.

    Args:
        request: The incoming user message.
        username: The authenticated username.

    Returns:
        StreamingResponse: A stream of AI-generated responses in JSON format.
    """
    if not hasattr(globals().get("graph"), "astream"):
        logger.error("LangGraph agent is not initialized")
        return StreamingResponse(
            content=[
                json.dumps(
                    {
                        "type": "text",
                        "content": "Error: LangGraph agent is not initialized\n",
                    }
                )
            ],
            media_type="application/json",
        )

    config = {"configurable": {"thread_id": request.session_id}}
    logger.info(
        f"Starting chat stream for user {username} with thread_id {config['configurable']['thread_id']}"
    )

    async def event_stream():
        # Regex to find Markdown image syntax with "sandbox:/file-ID"
        # It captures the alt text and the file ID
        image_file_id_pattern = re.compile(
            r"<IMAGE_GENERATED_FILE_ID>(.*?)</IMAGE_GENERATED_FILE_ID>"
        )
        openai_async_client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        last_message = None

        try:
            async for event in graph.astream(
                {"messages": [{"role": "user", "content": request.message}]},
                stream_mode="values",
                config=config,
            ):
                messages = event.get("messages", [])
                if messages:
                    last_message = messages[-1]

            if (
                last_message
                and isinstance(last_message, AIMessage)
                and last_message.content
            ):
                processed_text = last_message.content
                matches = list(image_file_id_pattern.finditer(processed_text))

                if matches:
                    # Process matches in order to ensure correct text segmentation
                    last_idx = 0
                    for match in matches:
                        file_id = match.group(1)

                        # Yield any text before the current image
                        if match.start() > last_idx:
                            text_before = processed_text[
                                last_idx : match.start()
                            ].strip()
                            if text_before:
                                yield json.dumps(
                                    {
                                        "type": "text",
                                        "content": text_before + "\n",
                                    }
                                )

                        logger.info(
                            f"Detected Markdown image with file ID: {file_id}. Attempting to fetch."
                        )
                        try:
                            image_data_response = (
                                await openai_async_client.files.content(file_id)
                            )
                            image_bytes = image_data_response.read()
                            if not image_bytes:
                                logger.error(
                                    f"No image data returned for file ID: {file_id}"
                                )
                                yield json.dumps(
                                    {
                                        "type": "text",
                                        "content": f"Error: No image data found for file ID {file_id}\n",
                                    }
                                )
                            else:
                                encoded_image = base64.b64encode(image_bytes).decode(
                                    "utf-8"
                                )
                                mime_type = "image/png"  # Assume PNG from Code Interpreter for now

                                logger.info(
                                    f"Fetched image for {file_id}. Size: {len(image_bytes)} bytes. Base64 length: {len(encoded_image)} chars."
                                )

                                try:
                                    yield json.dumps(
                                        {
                                            "type": "image",
                                            "content": encoded_image,
                                            "mime_type": mime_type,
                                            "alt_text": "",  # Include alt text for caption
                                        }
                                    ) + "\n"
                                except Exception as send_e:
                                    logger.error(
                                        f"Error sending image chunk for file {file_id}: {str(send_e)}",
                                        exc_info=True,
                                    )
                                    yield json.dumps(
                                        {
                                            "type": "text",
                                            "content": f"Error sending image chunk: {str(send_e)}\n",
                                        }
                                    )
                        except Exception as img_e:
                            logger.error(
                                f"Error fetching or encoding image file {file_id}: {str(img_e)}",
                                exc_info=True,
                            )
                            yield json.dumps(
                                {
                                    "type": "text",
                                    "content": f"Error fetching image: {str(img_e)}\n",
                                }
                            )

                        last_idx = match.end()

                    # Yield any remaining text after the last image
                    if last_idx < len(processed_text):
                        text_after = processed_text[last_idx:].strip()
                        if text_after:
                            yield json.dumps(
                                {
                                    "type": "text",
                                    "content": text_after + "\n",
                                }
                            )
                else:
                    # No Markdown image found, send as plain text
                    yield json.dumps(
                        {"type": "text", "content": last_message.content + "\n"}
                    )

        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            yield json.dumps({"type": "text", "content": f"Error: {str(e)}\n"})

    return StreamingResponse(event_stream(), media_type="application/json")


@app.post("/index")
async def index_documents(username: Annotated[str, Depends(get_current_username)]):
    """Trigger indexing of documents.

    Args:
        username: The authenticated username.

    Returns:
        dict: Status message indicating success or failure.

    Raises:
        HTTPException: If indexing fails.
    """
    try:
        await vector_store_manager.delete_all_documents()
        result = await vector_store_manager.index_documents()
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to index documents",
            )
        return {"message": f"Indexed {len(result)} documents successfully"}
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing documents: {str(e)}",
        )


@app.post("/login")
def login(request: LoginRequest):
    """Authenticate a user.

    Args:
        request: Login request containing username and password.

    Returns:
        dict: Login status and username.

    Raises:
        HTTPException: If authentication fails.
    """
    if db_manager.verify_user(request.username, request.password):
        session_id = generate_session_id(request.username)
        return {
            "message": "Login successful",
            "username": request.username,
            "session_id": session_id,
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )
