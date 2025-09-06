from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List
from dotenv import load_dotenv
from db_handler import logger, stored_data
import os

load_dotenv()

@tool
def internet_research_tool(query: str) -> str:
    try:
        logger.info("Looking for internet information regarding your query")
        search_tool_instance = DuckDuckGoSearchRun()
        result = search_tool_instance.run(query)
        logger.info("Web search tool successfully retrieved search results")
        return result
    except Exception as e:
        logger.exception("Failed to search the internet for your query")
        return f"No information collected regarding {query}"

@tool
def rag_tool(query: str) -> str:
    try:
        stored_data.invoke(query)
        logger.info("Successfully searched the vector db")
    except Exception as e:
        logger.exception("Failed to search the vector db")
        return "No relevant documents found"
    
def get_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        logger.info("Loaded groq API key")
    except Exception as e:
        logger.exception("Failed to load groq API key")
        raise
    try:
        model_name = "openai/gpt-oss-20b"
        llm = ChatGroq(
        model=model_name,
        temperature=0.7,
        reasoning_effort="medium"
        )
        logger.info(f"Successfully initialized the AI model: {model_name}")
        return llm
    except Exception as e:
        logger.exception(f"Failed to load the AI model: {model_name}")
        raise

@CrewBase
class Babynest():
    """Babynest crew"""
    def __init__(self):
        import os
        import yaml

        base_dir = os.path.dirname(__file__)
        config_dir = os.path.join(base_dir, "config")

        try:
            with open(os.path.join(config_dir, "agents.yaml"), "r") as f:
                self.agents_config = yaml.safe_load(f)
            with open(os.path.join(config_dir, "tasks.yaml"), "r") as f:
                self.tasks_config = yaml.safe_load(f)
            logger.info("Configuration files loaded successfully")
        except Exception as e:
            logger.warning("Failed to load config files: %s", e)
            self.agents_config = {}
            self.tasks_config = {}

        self.agents: List[BaseAgent]
        self.tasks: List[Task]

    
    @agent
    def maternal_health_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['maternal_health_researcher'], # type: ignore[index]
            verbose=True,
            tools=[internet_research_tool, rag_tool],
            llm=get_llm()
        )

    @agent
    def personalized_guidance(self) -> Agent:
        return Agent(
            config=self.agents_config['personalized_guidance'], # type: ignore[index]
            verbose=True,
            llm=get_llm(),
            tools=[rag_tool,internet_research_tool]
        )
    @agent
    def community_testimonials(self) -> Agent:
        return Agent(
            config=self.agents_config['community_testimonials'], # type: ignore[index]
            verbose=True,
            llm=get_llm(),
            tools=[rag_tool,internet_research_tool]
        )
    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'], # type: ignore[index]
            verbose=True,
            llm=get_llm(),
            # tools=[rag_tool,internet_research_tool]
        )
    @agent
    def moderator(self) -> Agent:
        return Agent(
            config=self.agents_config['moderator'], # type: ignore[index]
            verbose=True,
            llm=get_llm(),
            # tools=[rag_tool,internet_research_tool]
        )
    

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def ai_support_task(self) -> Task:
        return Task(
            config=self.tasks_config['health_support'], # type: ignore[index]
            agent=[
                self.maternal_health_researcher(),
                self.personalized_guidance(),
                self.community_testimonials(),
                self.moderator(),
                self.summarizer()
            ]
        )

    @task
    def postpartum_support(self) -> Task:
        return Task(
            config=self.tasks_config['postpartum_support'], # type: ignore[index],
            agent = [
                self.personalized_guidance(),
                self.moderator(),
                self.maternal_health_researcher(),
                self.summarizer()
            ]
            
        )
    
    @task
    def testimonial_finder(self) -> Task:
        return Task(
            config=self.tasks_config['testimonial_support'], # type: ignore[index],
            agent = [
                self.community_testimonials(),
                self.moderator(),
                self.maternal_health_researcher(),
            ]
            
        )
    @task
    def final_summary(self) -> Task:
        return Task(
            config=self.tasks_config["final_summary"],
            agent=self.summarizer(),
            depends_on=[
                self.ai_support_task(),
                self.postpartum_support(),
                self.testimonial_finder()
            ]
        )

    

    @crew
    def crew(self) -> Crew:
        """Creates the Babynest crew"""
        
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=[
                self.health_support(),
                self.postpartum_support(),
                self.testimonial_finder(),
                self.final_summary(),   # ✅ always at the end
            ], # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
