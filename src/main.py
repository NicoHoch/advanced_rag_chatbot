import os
from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from langchain_core.messages import AIMessage
import json
import re
import base64
import openai
from src.models.login import LoginRequest
from src.services.postgres_service import PostgresManager
from src.services.langgraph_service import LanggraphAgentManager
from src.services.vector_store_service import VectorStoreManager
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse

from src.utils.session import generate_session_id
from fastapi import UploadFile, File

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

db_manager = PostgresManager()
vector_store_manager = VectorStoreManager(db_manager.db_uri)
langgraph_manager = LanggraphAgentManager(db_manager.db_uri)
graph = None  # Global graph variable to store agent_executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context for FastAPI app."""
    global graph
    try:
        logger.info("Initializing database and LangGraph agent")
        db_manager.create_tables()
        # await langgraph_manager.index_documents() # Commented out for faster startup
        graph = await langgraph_manager.create_graph()
        if graph is None:
            raise ValueError(
                "Failed to initialize LangGraph agent: agent_executor is None"
            )
        logger.info("Startup completed successfully")
        yield
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        try:
            logger.info("Closing database connection")
            db_manager.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")


app = FastAPI(lifespan=lifespan)
security = HTTPBasic()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> str:
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
    message: str
    session_id: str


@app.post("/session_id")
def get_session_id(username: Annotated[str, Depends(get_current_username)]):
    session_id = generate_session_id(username)
    return {"session_id": session_id}


@app.get("/rag_sources")
async def get_rag_sources(username: Annotated[str, Depends(get_current_username)]):
    """
    Returns the list of RAG source file names currently indexed in the database.
    """
    try:
        sources = await db_manager.get_all_source_filenames()
        return {"sources": sources}
    except Exception as e:
        logger.error(f"Error fetching RAG sources: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching RAG sources: {str(e)}",
        )


@app.post("/upload_files")
async def upload_files(
    username: Annotated[str, Depends(get_current_username)],
    file: UploadFile = File(...),
):
    """
    Uploads a file and indexes it as a RAG source.
    """

    try:
        filename = file.filename
        content = await file.read()
        # Save file temporarily
        temp_path = f"/tmp/{filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        # Index the document
        result = await vector_store_manager.index_document_from_file(
            temp_path, filename
        )
        os.remove(temp_path)
        if result:
            return {"message": f"Uploaded and indexed: {filename}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to index {filename}",
            )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}",
        )


@app.post("/chat")
async def chat(
    request: MessageRequest, username: Annotated[str, Depends(get_current_username)]
):
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
                    last_idx = 0
                    for match in matches:
                        file_id = match.group(1)
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
                                mime_type = "image/png"
                                logger.info(
                                    f"Fetched image for {file_id}. Size: {len(image_bytes)} bytes. Base64 length: {len(encoded_image)} chars."
                                )
                                try:
                                    yield json.dumps(
                                        {
                                            "type": "image",
                                            "content": encoded_image,
                                            "mime_type": mime_type,
                                            "alt_text": "",
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
                    yield json.dumps(
                        {"type": "text", "content": last_message.content + "\n"}
                    )

        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            yield json.dumps({"type": "text", "content": f"Error: {str(e)}\n"})

    return StreamingResponse(event_stream(), media_type="application/json")


@app.post("/index")
async def index_documents(username: Annotated[str, Depends(get_current_username)]):
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
