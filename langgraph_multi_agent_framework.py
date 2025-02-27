from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, Graph
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from agentFrameworks import call_crew_agent, call_autogen_agent
from langchain.schema import HumanMessage, AIMessage, SystemMessage

model_name = "gpt-4o-mini"
# Define the state for our graph


class AgentState(MessagesState):
    """State for the agent system"""
    pass


# reAct agent
model = ChatOpenAI(
    model=model_name,
    verbose=True,
    timeout=30,  # Add a 30-second timeout
    max_retries=2  # Limit retries
)

# Define a simple tool for the agent


@tool
def add(num1: int, num2: int) -> int:
    """
    Adds two integers and returns the sum.

    Args:
        num1 (int): The first number.
        num2 (int): The second number.

    Returns:
        int: The sum of num1 and num2.
    """

    data = "debugger"
    # Simulating search results
    return num1 + num2


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


@tool
def get_user_info(phoneNumber: str) -> str:
    """
    Searches for user information based on the user phone number.

    Args:
        phoneNumber:  users phone number

    Returns:
        user results as a string
    """

    data = "debugger"
    # Simulating search results
    return f"this user name is bumindu yasith and the user id is 123456"


# Create React agent with the tool
tools = [add, multiply, get_user_info]
react_agent = create_react_agent(
    model,
    tools=tools,  # check this by adding a tool to the list
    prompt=("""
            You work as an operator in the help center for an internet service provider. When a user contacts you for assistance, 
            make sure to first ask for their phone number if they haven't provided it. Once you have the phone number, use it to retrieve the user's information. 
            and return the user information as a strucutred message. dont include any other text just the user information as json
            if you cant get the user information let him know that you cant find the user information thats it nothing else.
            """
            )
)


# Define the agent node function
def call_react_agent(state: AgentState):
    """Node function for the agent"""
    print("Starting call_react_agent...")

    # Get the messages from the state
    messages = state["messages"]

    # Invoke the agent
    result = react_agent.invoke({"messages": messages})

    # Update messages with agent's response
    state["messages"] = result["messages"]

    print("Finished call_react_agent")
    # Return the updated state and move to END
    return {"messages": state["messages"]}

# Create the graph


builder = Graph()

# Add nodes
builder.add_node("UserQueryProcessor", call_react_agent)
builder.add_node("UsageAnalyzer", call_crew_agent)
builder.add_node("RecommendationGenerator", call_autogen_agent)

# Add edges
builder.add_edge(START, "UserQueryProcessor")
builder.add_edge("UserQueryProcessor", "UsageAnalyzer")
builder.add_edge("UsageAnalyzer", "RecommendationGenerator")
builder.add_edge("RecommendationGenerator", END)

graph = builder.compile()
print("starting ............")
result = graph.invoke({"messages": [
    HumanMessage(
        content="seems my data limit is over could you please suggest me a good package my number is 0713945222")
]})

msg = result["messages"]
print(result["messages"][-1].content)
