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


@tool
def save(filename: str) -> str:
    """save current doc to txt file

    args:
    filename : name of file
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(document_content)
            print(f"Document has been saved to: {filename}")
            return f"Document has been saved successfully to '{filename}'."
        
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, save]

model = ChatOpenAI(model = 'gpt-4o').bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content =f"""
You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state['messages']:
        user_input = input("\nWhat would you like to do with the document? ")
        user_message = HumanMessage(content = user_input)
        
    else:
        user_input = input("\n What would you like to do with the document? ")    
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content = user_input)

    all_messages = [system_prompt] + List(state['messages']) + [user_message]

    response = model.invoke(all_messages)

    print(f'\n AI: {response.content}')
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": List(state['messages']) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the convo"""

    messages = state['messages']

    if not messages:
        return "continue"
    
    # look for most recent tool
    for message in reversed(messages):
        if(isinstance(message, ToolMessage) and
           "saved" in messages.content.lower() and
           "document" in message.content.lower()):
            return 'end' # goes to end node
    
    return "continue"


def print_messages(messages):
    """Func to print msg in readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", Toolnode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

