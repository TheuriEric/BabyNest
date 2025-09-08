from fastapi import FastAPI, HTTPException, Request
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from chat_models import ChatRequest, ChatResponse, SessionEndRequest
from components import retriever, AdaptiveConversation, session_memory, ValidationTool, ReasoningTool
from db_handler import text_splitter
from dotenv import load_dotenv
from crew import Babynest,get_llm,llm_clients
import logging
import os

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file = __name__.strip("__")
logger = logging.getLogger(file)

limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

try:
    os.environ["LANGSMITH_TRACING_V2"]="true"
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API)_KEY")
    os.environ["LANGSMITH_PROJECT"] = "BabyNest"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    logger.info("Successfully initialized Langsmith tracing")
except Exception as e:
    logger.exception("Failed to initialize langsmith tracing")

try:
    app = FastAPI(title="BabyNest",
                description="A pregnancy and postpartum platform backend",
                version="0.0.1")
    logger.info("Successfully initialized FastAPI app")
except Exception as e:
    logger.error("Failed to initialize the FastAPI backend")
    raise
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Handles rate limit exceptions and returns a custom JSON response.
    """
    return JSONResponse(
        status_code=429,
        content={"detail": "Sorry, you have exceeded the rate limit (5/minute)"}
    )

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

origins=[]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_headers=["*"],
    allow_credentials="True",
    allow_methods=["POST", "GET"]
)

llm = get_llm()
adaptive_convo = AdaptiveConversation(summarizing_llm=llm,)
crew_instance = Babynest().crew()
reasoning_tool = ReasoningTool(llm=llm)
validation_tool = ValidationTool(llm=llm)

async def route_query(user_request: str) -> str:
    """
    Intelligent router using a lightweight LLM to determine the best workflow.
    """

    router_prompt = ChatPromptTemplate.from_template(
        """
        You are a routing expert. Your task is to analyze the user's query and decide whether to route it to 'crewai' for health-related advice or 'langchain' for general chat. 
        Respond with only a single word: 'crewai' or 'langchain'.

        Examples:
        User: "What are common pregnancy symptoms?"
        Response: crewai

        User: "Hello, how are you today?"
        Response: langchain

        User: "{query}"
        Response: 
        """
    )
    router_chain = router_prompt | llm | StrOutputParser()
    try:
        decision = await router_chain.ainvoke({"query": user_request})
        return decision.strip().lower()
    except Exception as e:
        logger.warning(f"Router LLM failed, falling back to keyword matching: {e}")
        health_keywords = ["symptoms", "pain", "doctor", "swollen", "vomiting", "bleeding", "postpartum"]
        if any(keyword in user_request.lower() for keyword in health_keywords):
            return "crewai"
        return "langchain"

async def invoke_llm_with_fallback(chain, user_input, chat_history):
    """
    Invokes the primary LLM with a comprehensive fallback mechanism.
    """
    primary_llm_chain = chain

    fallback_llm_chain = chain.with_config(run_name="fallback", configurable={"llm": llm_clients["gemini"]})
    
    try:
        result = await primary_llm_chain.ainvoke({"user_input": user_input, "chat_history": chat_history})
        return result
    except Exception as e:
        logger.error(f"Primary LLM invocation failed. Falling back to secondary LLM: {e}")
        try:
            # Fallback to the secondary LLM (Gemini)
            result = await fallback_llm_chain.ainvoke({"user_input": user_input, "chat_history": chat_history})
            return result
        except Exception as fallback_e:
            logger.error(f"Fallback LLM also failed: {fallback_e}")
            raise HTTPException(status_code=500, detail="Failed to process request with all available models.")

async def mcp_pipeline(user_request: str, session_id: str = None):
    """
    Modular Conversational Pipeline (MCP):
    - Route query type
    - Retrieve knowledge if needed
    - Reason and validate
    - Generate response
    """
    if not session_id:
        session_id = str(uuid4())
        session_memory[session_id] = []
    elif session_id not in session_memory:
        session_memory[session_id] = []
        
    history = session_memory[session_id]
    formatted_history = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])

    route = await route_query(user_request)
    
    if route == "crewai":
        logger.info("Routing query to CrewAI for multi-agent collaboration.")
        try:
            raw_response = crew_instance.kickoff(inputs={'user_input': user_request})
        except Exception as e:
            logger.error(f"CrewAI execution failed: {e}")
            raise HTTPException(status_code=500, detail="CrewAI workflow failed.")
    
    else: 
        logger.info("Routing query to LangChain RAG chain for general chat.")

        retrieved_docs = retriever.get_documents(user_request)

        reasoning = await reasoning_tool.reason(
            user_input=user_request, 
            chat_history=formatted_history, 
            retrieved_docs=retrieved_docs
        )
        
        raw_response = await validation_tool.validate(reasoning)

    session_memory[session_id].append((user_request, raw_response))
    
    return {
        "session_id": session_id,
        "response": raw_response
    }

@app.get("/")
async def root():
    return {"message": "Successfully loaded the backend. Please visit /docs"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    return await mcp_pipeline(request.user_request, request.session_id)

@app.post("/api/external")
async def external_functions():
    # n8n part
    # Perhaps could be used to send out emails
    pass

@app.post("/api/end_session")
async def end_session(request: SessionEndRequest):
    """
    Summarizes the entire session and adds it to the persistent database.
    """
    session_id = request.session_id
    if session_id not in session_memory:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    conversation = session_memory[session_id]
    if not conversation:
        del session_memory[session_id]
        return {"message": "Session was empty. No data saved."}

    try:
        summary = adaptive_convo.summarize_conversation(conversation)
        adaptive_convo.add_convo_history_to_db(summary)
        
        # Clean up the in-memory cache
        del session_memory[session_id]
        
        return {"message": "Session summarized and saved successfully.", "summary": summary}
    except Exception as e:
        logger.exception("Failed to end session and save conversation.")
        raise HTTPException(status_code=500, detail="Failed to save session data.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)