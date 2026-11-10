from fastapi import FastAPI
from fastapi import APIRouter
from .schema import chat_payload


ai_agent_router = APIRouter(prefix="/ai")
app = FastAPI()

@ai_agent_router.post("/chat")
def initiate_chat(query: chat_payload):
    print(query.model_dump())
    return query



@ai_agent_router.get("/history")
def get_history(session_id: str):
    print("Session id has been recieved")
    return {"session_id": session_id, "history": []}
