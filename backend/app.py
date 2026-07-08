import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .assistant import get_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    context: list
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    answer, similar_items = get_answer(
        request.message,
        source="site",
        external_id=session_id,
        top_k=request.top_k,
    )

    return ChatResponse(answer=answer, context=list(similar_items), session_id=session_id)


@app.get("/health")
def health():
    return {"status": "ok"}