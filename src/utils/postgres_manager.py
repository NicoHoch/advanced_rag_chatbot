import psycopg2
from psycopg2.extras import execute_values
from typing import List, Tuple, Optional
from passlib.context import CryptContext
import urllib.parse
import os
from azure.identity import DefaultAzureCredential
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class PostgresManager:
    def __init__(self):
        """Initialize connection using Microsoft Entra ID authentication with token caching."""
        # Initialize credential once (shared across instances)
        if not hasattr(PostgresManager, "_credential"):
            PostgresManager._credential = DefaultAzureCredential()
        self.credential = PostgresManager._credential

        # Cache token and expiry
        if not hasattr(PostgresManager, "_token_cache"):
            PostgresManager._token_cache = {"token": None, "expiry": None}
        self.token_cache = PostgresManager._token_cache

        # Connect to database
        self._connect()

    def _get_token(self) -> str:
        """Get or refresh Entra ID token."""
        current_time = datetime.now(timezone.utc)
        if (
            self.token_cache["token"] is None
            or self.token_cache["expiry"] is None
            or current_time >= self.token_cache["expiry"]
        ):
            token_response = self.credential.get_token(
                "https://ossrdbms-aad.database.windows.net/.default"
            )
            self.token_cache["token"] = token_response.token
            self.token_cache["expiry"] = datetime.fromtimestamp(
                token_response.expires_on, timezone.utc
            )
        return self.token_cache["token"]

    def _connect(self):
        """Establish database connection."""
        # Read environment variables
        dbhost = os.environ.get("DBHOST", "your_dbhost")
        dbname = os.environ.get("DBNAME", "your_dbname")
        dbuser = urllib.parse.quote(os.environ.get("DBUSER", "your_dbuser"))
        sslmode = os.environ.get("SSLMODE", "require")

        # Get token
        password = self._get_token()

        # Construct connection URI
        self.db_uri = (
            f"postgresql://{dbuser}:{password}@{dbhost}/{dbname}?sslmode={sslmode}"
        )

        # Establish connection
        self.connection = psycopg2.connect(self.db_uri)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def create_tables(self):
        """Create users and documents tables and ensure admin user exists."""
        # Create users table
        users_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        """
        self.cursor.execute(users_query)

        # Create documents table
        documents_query = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(1536)
        );
        """
        self.cursor.execute(documents_query)

        # Check for admin user
        admin_username = os.environ.get("ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("ADMIN_PASSWORD", "default_admin_password")

        # Check if admin user exists
        query = """
        SELECT EXISTS (
            SELECT 1 FROM users WHERE username = %s
        );
        """
        self.cursor.execute(query, (admin_username,))
        admin_exists = self.cursor.fetchone()[0]

        if not admin_exists:
            # Insert admin user with hashed password
            password_hash = self.pwd_context.hash(admin_password)
            insert_query = """
            INSERT INTO users (username, password_hash)
            VALUES (%s, %s)
            RETURNING id;
            """
            self.cursor.execute(insert_query, (admin_username, password_hash))
            admin_id = self.cursor.fetchone()[0]
            print(f"Created admin user '{admin_username}' with ID {admin_id}")

    def insert_user(self, username: str, password: str) -> int:
        """Insert a new user with a hashed password."""
        password_hash = self.pwd_context.hash(password)
        query = """
        INSERT INTO users (username, password_hash)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (username, password_hash))
        return self.cursor.fetchone()[0]

    def verify_user(self, username: str, password: str) -> bool:
        """Verify if the username and password match."""
        query = """
        SELECT password_hash FROM users WHERE username = %s;
        """
        self.cursor.execute(query, (username,))
        result = self.cursor.fetchone()
        if result:
            password_hash = result[0]
            return self.pwd_context.verify(password, password_hash)
        return False

    def insert_document(self, content: str, embedding: List[float]) -> int:
        """Insert a document."""
        query = """
        INSERT INTO documents (content, embedding)
        VALUES (%s, %s)
        RETURNING id;
        """
        self.cursor.execute(query, (content, embedding))
        return self.cursor.fetchone()[0]

    def search_similar_documents(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """Search similar documents."""
        query = """
        SELECT id, content, embedding <-> %s AS distance
        FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s;
        """
        self.cursor.execute(query, (query_embedding, query_embedding, top_k))
        return self.cursor.fetchall()

    def close(self):
        """Close cursor and connection."""
        self.cursor.close()
        self.connection.close()
