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
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        "confirm access to weather information",
    )
    print("*******************starting get weather api call*******************")
    return "It might be cloudy in nyc {location}".format(location=location)


ag = create_react_agent(
    model, tools=[get_weather], prompt=(
        """you are a helpful assistant. greet user and reply to user question friendly. 
        if user ask to get weather information, use the tools you have.""")
)


def get_agent(state):
    result = ag.invoke({"messages": state["messages"]})
    return result


def finalized(agent_state: AgentState):
    """A function to handle finalization."""
    print("Finalized")
    print(agent_state)


builder = Graph()
builder.add_node("get_agent", get_agent)
builder.add_edge("get_agent", END)

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


def start_conversation(thread_id=None):

    conversation_config = config.copy()
    if thread_id:
        conversation_config["configurable"]["thread_id"] = thread_id

    with PostgresSaver.from_conn_string(DB_URI) as postgres_checkpointer:
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }

        postgres_checkpointer.setup()
        graph = builder.compile(checkpointer=postgres_checkpointer)
        msg = graph.get_state(conversation_config).values
        print(
            f"Starting a new conversation with thread_id: {conversation_config['configurable']['thread_id']}...")
        
        while True:
            msg = input("Enter your message: ")
            inputs = {"messages": [HumanMessage(content=msg)]}
            for s in graph.stream(
                    inputs, conversation_config, stream_mode="values", subgraphs=True):
                data = s[-1]
                messages = data.get('messages')
                # Check if message is an instance of HumanMessage
                if messages and isinstance(messages[-1], AIMessage):
                    print(messages[-1].content)


thread_id = "005"
start_conversation(thread_id)
