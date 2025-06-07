from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from src.models.login import LoginRequest
from src.utils.postgres_manager import PostgresManager
from src.utils.langgraph_manager import LanggraphManager
from fastapi.responses import StreamingResponse

app = FastAPI()

security = HTTPBasic()

# Initialize PostgresManager with Azure PostgreSQL credentials
db_manager = PostgresManager()
langgraph_manager = LanggraphManager(db_manager.db_uri)
graph = langgraph_manager.create_graph()

# Create tables and ensure admin user on startup
db_manager.create_tables()


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):
    """Authenticate user against the PostgreSQL database."""
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


@app.post("/chat")
def chat(
    request: MessageRequest, username: Annotated[str, Depends(get_current_username)]
):
    config = {"configurable": {"thread_id": "def234"}}

    def event_stream():
        latest_ai_message = None
        for event in graph.stream(
            {"messages": [{"role": "user", "content": request.message}]},
            stream_mode="values",
            config=config,
        ):
            for msg in event["messages"]:
                if type(msg).__name__ == "AIMessage" and msg.content != "":
                    latest_ai_message = msg.content
        if latest_ai_message:
            yield latest_ai_message + "\n"

    return StreamingResponse(event_stream(), media_type="text/markdown")


@app.post("/index")
def index_documents(username: Annotated[str, Depends(get_current_username)]):
    """Endpoint to trigger indexing of documents."""
    langgraph_manager.delete_all_documents()
    langgraph_manager.index_documents()
    return {"message": "Documents indexed successfully"}


@app.post("/login")
def login(request: LoginRequest):
    """Endpoint to authenticate a user."""
    if db_manager.verify_user(request.username, request.password):
        return {"message": "Login successful", "username": request.username}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.on_event("shutdown")
def shutdown_event():
    """Close database connection on shutdown."""
    db_manager.close()
