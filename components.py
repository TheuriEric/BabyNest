from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.documents import Document
from typing import List, Tuple

from db_handler import stored_data, logger, text_splitter

class RAGTool:
    def __init__(self, retriever_instance):
        self.retriever = retriever_instance

    def get_documents(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in docs)
    
retriever = RAGTool(stored_data)


class AdaptiveConversation:
    conversation_history: List[Tuple[str, str]] = []

    def summarize_conversation(history: List[Tuple[str:str]],summarizing_llm):
        logger.info("Summarizing entire conversation...")
        full_conversation = "\n".join([f"User: {q} \nAI: {a}" for q, a in history])

        summary_prompt = ChatPromptTemplate.from_template(
        "Please summarize the following conversation in a concise and neutral way. The summary should be in the third person.\n\nConversation:\n{conversation}"
    )

        summarization_chain = summary_prompt | summarizing_llm
        summary = summarization_chain.invoke({"conversation": full_conversation})
        
        return summary.content
    
    def add_convo_history_to_db(text: str):
        logger.info("Adding conversation history to vector database...")
        doc = Document(page_content=text)
        chunks = text_splitter.split_documents([doc])

        try: 
            stored_data.add_documents(chunks)
            logger.info("Successfully added conveersation history to vectordb")
        except Exception as e:
            logger.exception("Failed to add summary to vectordb memory")
            raise

class Chat:
    pass