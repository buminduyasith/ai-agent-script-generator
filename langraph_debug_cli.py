from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import Literal
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
)
from langgraph.graph import START, END, Graph, MessagesState, StateGraph
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from langgraph.types import interrupt
from typing import TypedDict, Annotated
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class AgentState(TypedDict):
    value: str


def agent(state: AgentState) -> dict:
    k = llm.invoke("can you multiply 5 * 6")
    return {"value": f"{k} func 1"}


# Create the StateGraph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("agent", agent)

# Set the entry point
builder.set_entry_point("agent")

# Add edges
builder.add_edge("agent", END)

# Compile the graph
graph = builder.compile()
