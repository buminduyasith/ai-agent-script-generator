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

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = MemorySaver()


class AgentState(MessagesState):
    """State for our agents."""
    pass


@tool
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    print("starting api call....................")
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        "confirm access to weather information",
    )
    if answer == "yes":
        print("get_weather...........................")
        if location.lower() in ["nyc", "new york"]:
            return "It might be cloudy in nyc"
        elif location.lower() in ["sf", "san francisco"]:
            return "It's always sunny in sf"
        else:
            raise AssertionError("Unknown Location")
    else:
        print("no permission........................")
        return "You don't have permission to access this information."


tools = [get_weather]

react_agent = create_react_agent(
    model, tools=tools, prompt=("""you are a helpful assistant. use only tools below to answer the user's question.""")
)


def finalized(agent_state: AgentState):
    """A function to handle finalization."""
    print("Finalized")
    print(agent_state)


builder = Graph()
builder.add_node("react_agent", react_agent)
builder.add_node("finalized", finalized)

builder.add_edge("react_agent", "finalized")
builder.add_edge("finalized", END)

builder.set_entry_point("react_agent")

config = {"configurable": {"thread_id": "43"}}
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

DB_URI = "postgresql://postgres:mysecretpassword@localhost:5432/langgraph_db?sslmode=disable"


def start_conversation():
    """Start a new conversation."""
    with PostgresSaver.from_conn_string(DB_URI) as postgres_checkpointer:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        postgres_checkpointer.setup()
        graph = builder.compile(checkpointer=postgres_checkpointer)
        print("Starting a new conversation...")
        print_stream(graph.stream(inputs, config, stream_mode="values"))

# answer = input(">>> ")

# print_stream(graph.stream(Command(resume=answer),
#              config, stream_mode="values"))

def resum_graph():
    DB_URI = "postgresql://postgres:mysecretpassword@localhost:5432/langgraph_db?sslmode=disable"
    with PostgresSaver.from_conn_string(DB_URI) as postgres_checkpointer:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        postgres_checkpointer.setup()
        graph = builder.compile(checkpointer=postgres_checkpointer)
        print_stream(graph.stream(Command(resume="yes"),
                    config, stream_mode="values"))


#start_conversation()
resum_graph()