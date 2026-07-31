from fastapi import APIRouter

from database import get_unreviewed_unknown_questions

router = APIRouter()


@router.get("/unknown-questions")
def unknown_questions():
    return {"unknown_questions": get_unreviewed_unknown_questions()}
