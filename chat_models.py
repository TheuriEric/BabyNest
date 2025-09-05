from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_request: str

class ChatResponse(BaseModel):
    response: str
