from fastapi import APIRouter

from database import get_session_messages

router = APIRouter()


@router.get("/session/{session_id}")
def get_session(session_id: str):
    return {"session_id": session_id, "messages": get_session_messages(session_id)}
