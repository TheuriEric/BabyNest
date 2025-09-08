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
    """Searches the internet for information based on a user's query. 
    This is useful for finding up-to-date information not present in the internal knowledge base."""
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
    """Retrieves relevant documents from the vector database based on a user's query. 
    Use this tool to get information from the internal knowledge base."""
    try:
        result = stored_data.invoke(query)
        logger.info("Successfully searched the vector db")
        return result if result else "No relevant documents found in the db"
    except Exception as e:
        logger.exception("Failed to search the vector db")
        return "No relevant documents found"
    
def get_llm():
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        logger.info("Loaded groq API key")
    except Exception as e:
        logger.exception("Failed to load groq API key")
        raise
    try:
        model_name = "groq/llama-3.3-70b-versatile"
        llm = ChatGroq(
            model_name=model_name,
            temperature=0.7,
            api_key=groq_api_key
        )
        logger.info(f"Successfully initialized the AI model: {model_name}")
        return llm
    except Exception as e:
        logger.exception(f"Failed to load the AI model: {model_name}")
        raise

@CrewBase
class Babynest:
    """Babynest crew"""

    def __init__(self):
        import os, yaml

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
            self.agents_config, self.tasks_config = {}, {}

    @agent
    def maternal_health_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config.get('maternal_health_researcher', {}),
            verbose=True,
            tools=[internet_research_tool, rag_tool],
            llm=get_llm()
        )

    @agent
    def personalized_guidance(self) -> Agent:
        return Agent(
            config=self.agents_config.get('personalized_guidance', {}),
            verbose=True,
            llm=get_llm(),
            tools=[rag_tool, internet_research_tool]
        )

    @agent
    def community_testimonials(self) -> Agent:
        return Agent(
            config=self.agents_config.get('community_testimonials', {}),
            verbose=True,
            llm=get_llm(),
            tools=[rag_tool, internet_research_tool]
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config.get('summarizer', {}),
            verbose=True,
            llm=get_llm()
        )

    @agent
    def moderator(self) -> Agent:
        return Agent(
            config=self.agents_config.get('moderator', {}),
            verbose=True,
            llm=get_llm()
        )

    @task
    def ai_support_task(self) -> Task:
        return Task(
            config=self.tasks_config.get('health_support', {})
        )

    @task
    def postpartum_support(self) -> Task:
        return Task(
            config=self.tasks_config.get('postpartum_support', {})
        )
    
    @task
    def testimonial_finder(self) -> Task:
        return Task(
            config=self.tasks_config.get('testimonial_support', {})
        )

    @task
    def final_summary(self) -> Task:
        # Handle the agent field properly since it might be a list in your YAML
        task_config = self.tasks_config.get('final_summary', {}).copy()
        if 'agent' in task_config and isinstance(task_config['agent'], list):
            task_config['agent'] = task_config['agent'][0]  # Take first agent from list
        return Task(
            config=task_config
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Babynest crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )