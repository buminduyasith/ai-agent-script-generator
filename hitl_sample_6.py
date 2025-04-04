from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import Literal
import os
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage
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
def post_weather(location: str):
    """Use this to get weather information from a given location."""
    print("starting post weather api call")
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        "confirm access to weather information",
    )
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
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        "confirm access to weather information",
    )
    print("*******************starting get weather api call*******************")
    return "It might be cloudy in nyc {location}".format(location=location)


post_agent = create_react_agent(
    model, tools=[post_weather], prompt=(
        """you are a helpful assistant. use only tools below to answer the user's question.""")
)

ag = create_react_agent(
    model, tools=[get_weather], prompt=(
        """you are a helpful assistant. greet user and reply to user question friendly. 
        if user ask to get weather information, use the tools you have.""")
)
def get_agent(state):
    msg =state['messages'][-1][1]
    formatted_input = {"messages": [HumanMessage(content=msg)]}
    data = ag.invoke(formatted_input)
    


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



def print_stream(stream):
    """A utility to pretty print the stream."""
    for s in stream:
        if s[1] == "messages":
            if isinstance(s[-1][0], AIMessageChunk):
                content = s[-1][0].content
                print(content)


DB_URI = "postgresql://postgres:mysecretpassword@localhost:5432/langgraph_db?sslmode=disable"


def start_conversation(msg: str, thread_id=None):
    """Start a new conversation.

    Args:
        thread_id: Optional thread ID to use for the conversation.
    """
    # Use the provided thread_id if available, otherwise use the default
    conversation_config = config.copy()
    if thread_id:
        conversation_config["configurable"]["thread_id"] = thread_id
        
    inputs = {"messages": [("user", msg)]}
    with PostgresSaver.from_conn_string(DB_URI) as postgres_checkpointer:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        postgres_checkpointer.setup()
        graph = builder.compile(checkpointer=postgres_checkpointer)
        print(
            f"Starting a new conversation with thread_id: {conversation_config['configurable']['thread_id']}...")
        for s in graph.stream(
            inputs, conversation_config, stream_mode="values", subgraphs=True):
            if (s[1] == "messages"):
                content = s[1][-1]

# answer = input(">>> ")

# print_stream(graph.stream(Command(resume=answer),
#              config, stream_mode="values"))


def resum_graph(thread_id=None):
    """Resume a conversation.

    Args:
        thread_id: Optional thread ID to use for the conversation.
    """
    # Use the provided thread_id if available, otherwise use the default
    conversation_config = config.copy()
    if thread_id:
        conversation_config["configurable"]["thread_id"] = thread_id

    DB_URI = "postgresql://postgres:mysecretpassword@localhost:5432/langgraph_db?sslmode=disable"
    with PostgresSaver.from_conn_string(DB_URI) as postgres_checkpointer:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        postgres_checkpointer.setup()
        graph = builder.compile(checkpointer=postgres_checkpointer)
        state = graph.get_state(conversation_config).values
        next =graph.get_state(conversation_config).next
        print(
            f"Resuming conversation with thread_id: {conversation_config['configurable']['thread_id']}...")
        print_stream(graph.stream(Command(resume="yes"),
                                  conversation_config, stream_mode="values"))


# poetry run python .\hitl_sample_6.py sc 10 (10 -> thread_id)
# poetry run python .\hitl_sample_6.py rs 10 (10 -> thread_id)

# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) > 1:
#         command = sys.argv[1]
#         # Check if a thread_id is provided as the third argument
#         thread_id = sys.argv[2] if len(sys.argv) > 2 else None

#         if command == "sc":
#             start_conversation(thread_id)
#         elif command == "rs":
#             resum_graph(thread_id)
#         else:
#             print("Usage: python hitl_sample_6.py [sc|rs] [thread_id]")
#     else:
#         start_conversation(thread_id)
#         print("Usage: python hitl_sample_6.py [sc|rs] [thread_id]")

while True:
    thread_id = "003"
    command = input("Enter your message: ")
    if command != "yes":
        start_conversation(command, thread_id)
    else:
        resum_graph(thread_id)