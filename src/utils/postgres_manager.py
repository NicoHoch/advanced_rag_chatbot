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
        # Use a class-level credential and token cache for efficiency
        if not hasattr(PostgresManager, "_credential"):
            PostgresManager._credential = DefaultAzureCredential()
        self.credential = PostgresManager._credential

        if not hasattr(PostgresManager, "_token_cache"):
            PostgresManager._token_cache = {"token": None, "expiry": None}
        self.token_cache = PostgresManager._token_cache

        self.connection = None
        self.cursor = None
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
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
        """Establish database connection using environment variables and Entra ID token."""
        dbhost = os.environ.get("DBHOST")
        dbname = os.environ.get("DBNAME")
        dbuser = urllib.parse.quote(os.environ.get("DBUSER", ""))
        sslmode = os.environ.get("SSLMODE", "require")

        if not all([dbhost, dbname, dbuser]):
            raise ValueError(
                "Database connection environment variables are not set properly."
            )

        password = self._get_token()
        self.db_uri = (
            f"postgresql://{dbuser}:{password}@{dbhost}/{dbname}?sslmode={sslmode}"
        )

        if self.connection:
            try:
                self.cursor.close()
                self.connection.close()
            except Exception:
                pass

        self.connection = psycopg2.connect(self.db_uri)
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()

    def _ensure_connection(self):
        """Ensure the database connection is alive, reconnect if needed."""
        if not self.connection or self.connection.closed != 0:
            self._connect()

    def create_tables(self):
        """Create users and documents tables and ensure admin user exists."""
        self._ensure_connection()
        users_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        );
        """
        self.cursor.execute(users_query)

        documents_query = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding VECTOR(1536)
        );
        """
        self.cursor.execute(documents_query)

        admin_username = os.environ.get("ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("ADMIN_PASSWORD", "default_admin_password")

        query = "SELECT EXISTS (SELECT 1 FROM users WHERE username = %s);"
        self.cursor.execute(query, (admin_username,))
        admin_exists = self.cursor.fetchone()[0]

        if not admin_exists:
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
        self._ensure_connection()
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
        self._ensure_connection()
        query = "SELECT password_hash FROM users WHERE username = %s;"
        self.cursor.execute(query, (username,))
        result = self.cursor.fetchone()
        if result:
            password_hash = result[0]
            return self.pwd_context.verify(password, password_hash)
        return False

    def close(self):
        """Close cursor and connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
