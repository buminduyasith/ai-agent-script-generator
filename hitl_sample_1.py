from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, END, Graph, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
)
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()

class AgentState(MessagesState):
    """State for our agents."""
    pass


# Initialize the model (using your desired model; here we use "gpt-4o-mini")
model_name = "gpt-4o-mini"
model = ChatOpenAI(
    model=model_name,
    verbose=True,
    timeout=30,   # 30-second timeout
    max_retries=2  # Limit retries
)

# ------------------------------------------------------------------------------
# Define the tools
# ------------------------------------------------------------------------------


@tool
def get_data(query: str) -> str:
    """
    Simulates an API GET request to retrieve data based on the query.

    Args:
        query (str): The query to fetch data for.

    Returns:
        str: A JSON string representing the retrieved data.
    """
    # In a real scenario, you might perform a requests.get() call here.
    # For this example, we simulate a response.
    return '{"data": {"product": "Widget", "price": 19.99}}'


@tool
def post_data(payload: str) -> str:
    """
    Simulates an API POST request to send data.

    Args:
        payload (str): The data payload to post.

    Returns:
        str: A JSON string representing the API response.
    """
    answer = interrupt(
        # This value will be sent to the client
        # as part of the interrupt information.
        "confirm access to weather information",
    )
    # In a real scenario, you might perform a requests.post() call here.
    # For this example, we simulate a response.
    print(f"Posting data: {payload}")
    return '{"status": "success", "message": "Data posted successfully"}'

# ------------------------------------------------------------------------------
# Create the reAct agents
# ------------------------------------------------------------------------------


# Agent 1: Data Retrieval Agent using the GET-like tool
get_data_agent = create_react_agent(
    model,
    tools=[get_data],
    prompt=(
        """
You are a data retrieval operator. When given a query, use the get_data tool to fetch data from an external API.
Return only the retrieved data as a JSON structure. If no data is found, return an appropriate error message.
        """
    )
)

# Agent 2: Data Submission Agent using the POST-like tool
post_data_agent = create_react_agent(
    model,
    tools=[post_data],
    prompt=(
        """
You are a data submission operator. When provided with data, use the post_data tool to send a POST request to an external API.
Return only the API's response as a JSON structure. If the submission fails, return an appropriate error message.
        """
    )
)

# ------------------------------------------------------------------------------
# Define node functions to call each agent in the graph
# ------------------------------------------------------------------------------


def call_get_data_agent(state: AgentState):
    print("Starting call_get_data_agent...")
    messages = state["messages"]
    # Invoke the first agent to retrieve data using the get_data tool
    result = get_data_agent.invoke({"messages": messages})
    state["messages"] = result["messages"]
    print("Finished call_get_data_agent")
    return {"messages": state["messages"]}


def call_post_data_agent(state: AgentState):
    print("Starting call_post_data_agent...")
    messages = state["messages"]
    # Invoke the second agent to send the retrieved data using the post_data tool
    result = post_data_agent.invoke({"messages": messages})
    state["messages"] = result["messages"]
    print("Finished call_post_data_agent")
    return {"messages": state["messages"]}

# ------------------------------------------------------------------------------
# Build the execution graph
# ------------------------------------------------------------------------------


builder = Graph()
# Add nodes for each agent's function
builder.add_node("DataRetrieval", call_get_data_agent)
builder.add_node("DataSubmission", call_post_data_agent)

# Define the flow: START -> DataRetrieval -> DataSubmission -> END
builder.add_edge(START, "DataRetrieval")
builder.add_edge("DataRetrieval", "DataSubmission")
builder.add_edge("DataSubmission", END)

# Compile the graph
graph = builder.compile(checkpointer=memory)

# ------------------------------------------------------------------------------
# Execute the graph
# ------------------------------------------------------------------------------

print("Starting the graph execution...")

# Begin with an initial message that includes a query.
# For example: "Please fetch data for product id 101"

config = {"configurable": {"thread_id": "42"}}

result = graph.invoke({"messages": [
    HumanMessage(
        content="Please fetch data for product id 101 and create a new product with the same data")
]}, config=config)

# # Print the final output from the graph (response from the POST request)
# final_msg = result["messages"][-1].content
# print("Final Output:")
# print(final_msg)

print(result)

# for m in result["messages"]:
#     print(m)


answer = input(">>> ")
graph.invoke(Command(resume=answer), config, stream_mode="values")
