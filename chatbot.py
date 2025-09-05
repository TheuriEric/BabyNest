from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from chat_models import ChatRequest, ChatResponse
from components import retriever
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file = __name__.strip("__")
logger = logging.getLogger(file)

try:
    app = FastAPI(title="BabyNest",
                description="A pregnancy and postpartum platform backend",
                version="0.0.1")
    logger.info("Successfully initialized FastAPI app")
except Exception as e:
    logger.error("Failed to initialize the FastAPI backend")
    raise

origins=[]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_headers=["*"],
    allow_credentials="True",
    allow_methods=["POST", "GET"]
)
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)


@app.get("/")
async def root():
    return {"message": "Successfully loaded the backend. Please visit /docs"}


@app.post("/api/chat") # This will be the main chat function
async def chat():
    prompt_template = ChatPromptTemplate.from_template(""
    """
You are the main chat assistant for BabyNest. Here you handle all general chat features that are non-health related. You answer general queries in a friendly tone and in simple terms. Aim for accuracy in understanding using the simplest method. Any health related features are handled by

""")

@app.post("/api/health")
async def assistant(request: ChatRequest):
    # Symptom logging/analysis
    # Daily updates
    # Motivation
    # Track baby development weekly
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an intelligent, African-focused pregnancy and postpartum health assistant. You are to provide the ussers with accurate medical information and personalized tips to the best of your abilities. Be empathetic, humble and use simple language for easier understanding.
        Use the information provided to answer the users' queries as well as information from internet sources. You mainly answer pregnancy/motherhood related items. Any query outside this context you answer but provide a disclaimer that you only answer questions pertaining motherhood. Determine the tone and exact desire of the user so as to answer correctly and without errors. Be simple and answer using optimum number of words. Use the most natural number of words in a similar way a doctor would answer. Look into how doctor-patient conversations are conducted and follow the same Ensure you are conversational. Ask for more information from the user where there is ambiguity or where needed to increase efficiency and accuracy

        User query: {user_input}
        Content: {content}

        """
    )
    # messages = prompt_template.format_messages(user_input=request.user_request)
    # response = llm.invoke(messages)
    # formatted_response = response.content if hasattr(response, "content") else str(response)
    return {
        "user_input": RunnablePassthrough(),
        "content" : retriever
    } | prompt_template | llm | StrOutputParser()

@app.post("/api/external")
async def external_functions():
    # n8n part
    # Perhaps could be used to send out emails
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)