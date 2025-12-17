from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_core import ToolMessage
from langchain_core.tools import tool
from langchain.graph.message import add_messages
from langraph.prebuilt import Toolnode


load_dotenv()

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """update document"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"