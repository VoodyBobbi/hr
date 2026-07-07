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


class ChatResponse(BaseModel):
    answer: str
    context: list


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer, similar_items = get_answer(request.message, top_k=request.top_k)
    return ChatResponse(answer=answer, context=list(similar_items))


@app.get("/health")
def health():
    return {"status": "ok"}