from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_request: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

class SessionEndRequest(BaseModel):
    session_id: str