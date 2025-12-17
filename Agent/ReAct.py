from typing import TypedDict, List, Union, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_core import ToolMessage
from langchain_core import SystemMessage
from langchain_core.tools import tool
from langchain.graph.message import add_messages
from langraph.prebuilt import Toolnode




load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b:int):

    return a + b

tools = [add]

model = ChatOpenAI(model= 'gpt-4o').bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content = 
        "You are my Ai assistant, please answer my query to the best of your ability"
    )
    response = model.invoke([system_prompt])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)