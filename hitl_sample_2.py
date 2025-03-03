from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from typing import Literal
import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# For this tutorial we will use custom tool that returns pre-defined values for weather in two cities (NYC & SF)


@tool
def get_weather(location: str):
    """Use this to get weather information from a given location."""
    if location.lower() in ["nyc", "new york"]:
        return "It might be cloudy in nyc"
    elif location.lower() in ["sf", "san francisco"]:
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown Location")


@tool
def multiply(num1: int, num2: int) -> int:
    """
    Multiplies two integers and returns the product.

    Args:
        num1 (int): The first number.
        num2 (int): The second number.

    Returns:
        int: The product of num1 and num2.
    """

    return num1 * num2


tools = [get_weather, multiply]

# We need a checkpointer to enable human-in-the-loop patterns

memory = MemorySaver()

# Define the graph

graph = create_react_agent(
    model, tools=tools, interrupt_before=["tools"], checkpointer=memory
)

graph2 = create_react_agent(
    model, tools=tools, interrupt_before=['get_weather'], checkpointer=memory
)


def print_stream(stream):
    """A utility to pretty print the stream."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


config = {"configurable": {"thread_id": "42"}}
# inputs = {"messages": [("user", "what is the weather in new york")]}
inputs = {"messages": [("user", "what 5*6")]}

print_stream(graph.stream(inputs, config, stream_mode="values"))

answer = input(">>> ")

print_stream(graph.stream(None, config, stream_mode="values"))
