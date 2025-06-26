import uuid


def generate_session_id(username: str) -> str:
    return f"chat_{username}_{uuid.uuid4()}"
