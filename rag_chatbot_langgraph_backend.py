"""
This is a simple chatbot backend that uses LangGraph to create a RAG (Retrieval-Augmented Generation) chatbot.
It ingests PDF documents, splits them into chunks, creates embeddings, and uses them to answer user queries.

"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()


load_dotenv()

#1st
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Ingest pdfs
_RETRIEVER: Optional[Any] = None

# ===================== PDF INGEST =====================
def ingest_pdf(pdf_path: str, thread_id: Optional[str] = None):
    global _RETRIEVER

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    _RETRIEVER = vectorstore.as_retriever(search_kwargs={"k": 4})

    print(f"[INGEST] Pages: {len(docs)} | Chunks: {len(chunks)}")



#Graph

#state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_docs: list[str]


#nodes
def rag_node(s : ChatState) -> ChatState : 
  if _RETRIEVER is None:
        return {
            "retrieved_docs": [
                "No document has been uploaded yet. Please upload a PDF first."
            ]
        }
  query = s['messages'][-1].content
  docs = _RETRIEVER.invoke(query)
  return {
        "retrieved_docs": [d.page_content for d in docs]
    }
  

# node

def chat_node(s: ChatState) -> ChatState:
    context = "\n\n".join(s["retrieved_docs"])

    system_message = SystemMessage(
        content=(
            "You are a conversational chatbot. "
            "Use the provided PDF context if needed. "
            "Otherwise, answer using the conversation history."
        )
    )

    
    messages_for_llm = (
        [system_message]
        + s["messages"]           
        + [
            SystemMessage(
                content=f"PDF Context:\n{context}"
            )
        ]
    )

    response = llm.invoke(messages_for_llm)

    return {
        "messages": [AIMessage(content=response.content)]
    }


# graph

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("rag", rag_node)

graph.add_edge(START, "rag")
graph.add_edge("rag", "chat_node")
graph.add_edge("chat_node", END)

checkpointer = InMemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ui + backend function
#it just takes input from ui and sends latest output to ui


def chatbot(query: str) -> str:
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "single_chat"},
                "callbacks": [langfuse_handler],}
    )
    return result["messages"][-1].content