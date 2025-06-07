from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from src.models.login import LoginRequest
from src.utils.postgres_manager import PostgresManager
from src.utils.langgraph_manager import LanggraphManager

app = FastAPI()

security = HTTPBasic()

# Initialize PostgresManager with Azure PostgreSQL credentials
db_manager = PostgresManager()
langgraph_manager = LanggraphManager(db_manager.db_uri)

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
    # TODO: setup graph and invoke it here
    return {"username": username, "message": request.message}


@app.post("/index")
def index_documents(username: Annotated[str, Depends(get_current_username)]):
    """Endpoint to trigger indexing of documents."""
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
