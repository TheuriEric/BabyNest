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
from components import retriever, AdaptiveConversation, session_memory
from db_handler import text_splitter
from dotenv import load_dotenv
from crew import Babynest,get_llm
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


@app.get("/")
async def root():
    return {"message": "Successfully loaded the backend. Please visit /docs"}


# @app.post("/api/chat") # This will be the main chat function
# async def chat():
#     prompt_template = ChatPromptTemplate.from_template(""
#     """
# You are the main chat assistant for BabyNest. Here you handle all general chat features that are non-health related. You answer general queries in a friendly tone and in simple terms. Aim for accuracy in understanding using the simplest method. Any health related features are handled by

# """)
@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Determine which workflow to use based on the user's query
    route = route_query(request.user_request)

    if route == "crewai":
        try:
            logger.info("Routing query to CrewAI for multi-agent collaboration.")
            # The .kickoff() method starts the CrewAI process
            result = crew_instance.kickoff(inputs={'user_input': request.user_request}) # Use a similar input key as your LangChain chain
            
            return {
                "session_id": request.session_id or str(uuid4()),
                "response": result
            }
        except Exception as e:
            logger.error(f"CrewAI execution failed: {e}")
            raise HTTPException(status_code=500, detail="CrewAI workflow failed.")
    else: # Default to LangChain RAG
        logger.info("Routing query to LangChain RAG chain for general chat.")
        
        # This is your existing LangChain RAG logic
        if not request.session_id:
            session_id = str(uuid4())
            session_memory[session_id] = []
        else:
            session_id = request.session_id
        
        if session_id not in session_memory:
            session_memory[session_id] = []
            
        history = session_memory[session_id]
        formatted_history = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])
        
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are the main chat assistant for BabyNest. Here you handle all general chat features that are non-health related. You answer general queries in a friendly tone and in simple terms. Aim for accuracy in understanding using the simplest method. Any health related features are handled by
            
            User query: {user_input}
            Chat history: {chat_history}
            """
        )
        chain = (
            {
                "user_input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "content" : RunnableLambda(lambda x: retriever.get_documents(x["user_input"]))
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )

        result = chain.invoke({"user_input": request.user_request, "chat_history": formatted_history})
        
        session_memory[session_id].append((request.user_request, result))
        
        return {
            "session_id": session_id,
            "response": result
        }


@app.post("/api/health")
async def assistant(request: ChatRequest):
    # Symptom logging/analysis
    # Daily updates
    # Motivation
    # Track baby development weekly
    if not request.session_id:
        session_id = str(uuid4())
        session_memory[session_id] = []
    else:
        session_id = request.session_id
    
    if session_id not in session_memory:
        session_memory[session_id] = []
        
    history = session_memory[session_id]
    formatted_history = "\n".join([f"User: {q}\nAI: {a}" for q, a in history])
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an intelligent, African-focused pregnancy and postpartum health assistant. You are to provide the ussers with accurate medical information and personalized tips to the best of your abilities. Be empathetic, humble and use simple language for easier understanding.
        Use the information provided to answer the users' queries as well as information from internet sources. You mainly answer pregnancy/motherhood related items. Any query outside this context you answer but provide a disclaimer that you only answer questions pertaining motherhood. Determine the tone and exact desire of the user so as to answer correctly and without errors. Be simple and answer using optimum number of words. Use the most natural number of words in a similar way a doctor would answer. Look into how doctor-patient conversations are conducted and follow the same Ensure you are conversational. Ask for more information from the user where there is ambiguity or where needed to increase efficiency and accuracy.


        User query: {user_input}
        Content: {content}

        """
    )
    chain = {
        "user_input": RunnablePassthrough(),
        "chat_history": RunnablePassthrough(),
        "content" : RunnableLambda(lambda x: retriever.get_documents(x["user_input"]))
    } | prompt_template | llm | StrOutputParser()

    result = chain.invoke({"user_input": request.user_request, "chat_history": formatted_history})

    session_memory[session_id].append((request.user_request, result))

    return {
        "session_id": session_id,
        "response": result
    }

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