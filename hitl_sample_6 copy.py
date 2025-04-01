from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import Literal
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
)
from langgraph.graph import START, END, Graph, MessagesState
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from langgraph.types import interrupt
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0,
                   )


class AgentState(MessagesState):
    """State for our agents."""
    pass


@tool
def post_weather(location: str):
    """Use this to get weather information from a given location."""
    print("starting post weather api call")
    answer = "yes"
    if answer == "yes":
        print("post request...........................")
        if location.lower() in ["nyc", "new york"]:
            return "It might be cloudy in nyc"
        elif location.lower() in ["sf", "san francisco"]:
            return "It's always sunny in sf"
        else:
            raise AssertionError("Unknown Location")
    else:
        print("no permission........................")
        return "You don't have permission to access this information."


@tool
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    print("*******************starting get weather api call*******************")
    return "It might be cloudy in nyc {location}".format(location=location)


post_agent = create_react_agent(
    model, tools=[post_weather], prompt=(
        """you are a helpful assistant. use only tools below to answer the user's question.""")
)


get_agent = create_react_agent(
    model, tools=[get_weather], prompt=(
        """you are a helpful assistant. use only tools below to answer the user's question""")
)


def finalized(agent_state: AgentState):
    """A function to handle finalization."""
    print("Finalized")
    print(agent_state)


builder = Graph()
builder.add_node("get_agent", get_agent)
builder.add_node("post_agent", post_agent)
builder.add_node("finalized", finalized)

builder.add_edge("get_agent", "post_agent")
builder.add_edge("post_agent", "finalized")
builder.add_edge("finalized", END)

builder.set_entry_point("get_agent")

# Default config - will be overridden by command line argument if provided
config = {"configurable": {"thread_id": "44"}}
inputs = {"messages": [("user", "what is the weather in new york")]}


def print_stream(stream):
    """A utility to pretty print the stream."""
    for s in stream:
        if "messages" not in s:
            print(s)
            continue
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


graph = builder.compile()

# for s in graph.stream(inputs, stream_mode=['messages']):
#     data = s
#     print(s)
#     print("========================")

for s in graph.stream(inputs, stream_mode=['values']):
    data = s
    print(s)
    print("========================")
