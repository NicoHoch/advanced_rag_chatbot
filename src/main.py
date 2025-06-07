from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
from langchain_core.messages import AIMessage
from src.models.login import LoginRequest
from src.utils.postgres_manager import PostgresManager
from src.utils.langgraph_manager import LanggraphManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()
security = HTTPBasic()

# Initialize PostgresManager and LanggraphManager
db_manager = PostgresManager()
langgraph_manager = LanggraphManager(db_manager.db_uri)
graph = None  # Global graph variable to store agent_executor


@app.on_event("startup")
async def startup_event():
    """Initialize database and LangGraph agent on startup."""
    try:
        logger.info("Initializing database and LangGraph agent")
        db_manager.create_tables()  # Create database tables
        await langgraph_manager.index_documents()  # Index documents (optional)
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


@app.post("/chat")
async def chat(
    request: MessageRequest, username: Annotated[str, Depends(get_current_username)]
):
    """Stream chat responses from the LangGraph agent.

    Args:
        request: The incoming user message.
        username: The authenticated username.

    Returns:
        StreamingResponse: A stream of AI-generated responses in text format.
    """
    if not hasattr(globals().get("graph"), "astream"):
        logger.error("LangGraph agent is not initialized")
        return StreamingResponse(
            content=["Error: LangGraph agent is not initialized\n"],
            media_type="text/plain",
        )

    config = {"configurable": {"thread_id": f"chat_{username}_{request.message[:10]}"}}
    logger.info(
        f"Starting chat stream for user {username} with thread_id {config['configurable']['thread_id']}"
    )

    async def event_stream():
        try:
            async for event in graph.astream(
                {"messages": [{"role": "user", "content": request.message}]},
                stream_mode="values",
                config=config,
            ):
                for msg in event.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content:
                        yield msg.content + "\n"
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            yield f"Error: {str(e)}\n"

    return StreamingResponse(event_stream(), media_type="text/plain")


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
        await langgraph_manager.delete_all_documents()
        result = await langgraph_manager.index_documents()
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
        return {"message": "Login successful", "username": request.username}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )
