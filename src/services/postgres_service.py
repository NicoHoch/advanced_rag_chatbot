from typing import List
from psycopg import Connection
from passlib.context import CryptContext
import urllib.parse
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class PostgresManager:
    def __init__(self):
        """Initialize connection using Microsoft Entra ID authentication with token caching."""

        self.connection = None
        self.cursor = None
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self._connect()

    def _get_token(self) -> str:

        credential = DefaultAzureCredential()
        token_response = credential.get_token(
            "https://ossrdbms-aad.database.windows.net/.default"
        )
        return token_response.token

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

        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        self.connection = Connection.connect(self.db_uri, **connection_kwargs)
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
            logger.info(f"Created admin user '{admin_username}' with ID {admin_id}")

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

    async def get_all_source_filenames(self) -> List[str]:
        """Get all filenames indexed in the vector store.

        Returns:
            List[str]: List of filenames.
        """
        try:
            self._ensure_connection()
            query = (
                "SELECT DISTINCT cmetadata->>'file_name' FROM langchain_pg_embedding "
                "WHERE cmetadata->>'file_name' IS NOT NULL AND cmetadata->>'file_name' <> '';"
            )
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            filenames = [row[0] for row in results]
            logger.info(f"Retrieved {len(filenames)} filenames from the database.")

            return filenames

        except Exception as e:
            logger.error(f"Error retrieving filenames: {str(e)}")
            return []
