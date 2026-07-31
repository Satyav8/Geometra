from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    query: str
    turn_number: int


class SourceChunk(BaseModel):
    chunk_id: str
    section: str
    text: str
    similarity_score: float


class EvaluationResult(BaseModel):
    metric_name: str
    passed: bool
    score: Optional[float]
    detail: str


class ChatResponse(BaseModel):
    session_id: str
    turn_number: int
    response: str
    sources: List[SourceChunk]
    confidence_level: str
    is_unknown_question: bool
    evaluation: List[EvaluationResult]
    response_latency_ms: int
    input_tokens: int
    output_tokens: int
    support_email: Optional[str] = None
