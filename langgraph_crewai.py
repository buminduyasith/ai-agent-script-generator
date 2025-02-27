from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, convert_to_openai_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
import os
from crewai import Agent, Task, Crew
import logging

from crewai.tools import tool

# Set up your OpenAI API key - replace with your actual key or use environment variable
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the language model
model_name = "gpt-4o-mini"
# Define the state for our graph


class AgentState(MessagesState):
    """State for the agent system"""
    pass

# crewai agents


@tool("search on internet")
def search_on_internet(query: str) -> str:
    """this is to search on internet this tool get the query as a parameter and search on interent and return
    the result as a string"""
    # Function logic here
    data = "debugger"
    return """
    An AI Agent is a software system that can perceive its environment, process information, and take actions to achieve specific goals. 
    It operates autonomously or semi-autonomously using techniques from artificial intelligence, such as machine learning, natural language processing, and reasoning.
    Key Characteristics of an AI Agent
    Perception – Gathers data from its environment using sensors, APIs, or databases.
    Reasoning & Decision-Making – Analyzes input and determines the best course of action.
    Learning – Adapts and improves its performance over time.
    Autonomy – Works with minimal human intervention.
    Interaction – Communicates with users, other agents, or external systems.
    Types of AI Agents
    Simple Reflex Agents – Respond to specific stimuli (e.g., rule-based bots).
    Model-Based Agents – Maintain internal models of the world to make decisions.
    Goal-Based Agents – Take actions to achieve predefined goals.
    Utility-Based Agents – Optimize actions based on utility functions.
    Learning Agents – Improve their behavior using experience (e.g., reinforcement learning).
    """


researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover accurate and relevant information about the topic and provide a concise summary",
    backstory="You are an expert at finding and summarizing information efficiently. You know when you have enough information to complete your task.",
    verbose=True,
    allow_delegation=False,
    tools=[search_on_internet],
    llm_model=model_name,
    # Add max iterations to prevent infinite loops
    max_iter=3
)

# Define tasks
research_task = Task(
    description="Research the given topic and provide key insights. Once you have gathered sufficient information, synthesize it into a comprehensive analysis with key points and facts. Do not keep searching once you have enough information.",
    agent=researcher,
    expected_output="A comprehensive analysis with key points and facts about AI Agents."
)

# Define crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True,
    # Add a specific process to make execution more predictable
    process="sequential"
)


def call_crew_agent(state: AgentState):
    """Node function for the agent"""
    print("Starting call_crew_agent...")

    # Get the messages from the state
    messages = state["messages"]

    # Extract the last human message content for the crew
    human_messages = [
        msg.content for msg in messages if isinstance(msg, HumanMessage)]
    topic = human_messages[-1] if human_messages else "AI Agents"
    print(f"CrewAI topic: {topic}")

    try:
        # Invoke the crew
        print("Invoking CrewAI...")
        result = crew.kickoff(inputs={'topic': topic})
        print(f"CrewAI result type: {type(result)}")

        # Add the crew's response to the messages
        if isinstance(result, str):
            crew_response = result
        else:
            # Handle other result types
            crew_response = str(result)

        print(f"CrewAI response (truncated): {crew_response[:100]}...")
        messages.append(AIMessage(content=f"CrewAI Result: {crew_response}"))
    except Exception as e:
        # Handle any exceptions
        error_message = f"Error running CrewAI: {str(e)}"
        messages.append(AIMessage(content=error_message))
        print(error_message)

    # Update messages in state
    state["messages"] = messages

    print("Finished call_crew_agent")
    # Return the updated state
    return {"messages": state["messages"]}


# reAct agent
model = ChatOpenAI(
    model=model_name,
    verbose=True,
    timeout=30,  # Add a 30-second timeout
    max_retries=2  # Limit retries
)

# Define a simple tool for the agent


@langchain_tool
def search_info(query: str) -> str:
    """
    Search for information on a given query.

    Args:
        query: The search query

    Returns:
        Search results as a string
    """

    data = "debugger"
    # Simulating search results
    return f"Here are the results for '{query}': This is simulated information about {query}."


# Create React agent with the tool
tools = [search_info]
react_agent = create_react_agent(
    model,
    tools=[search_info],  # check this by adding a tool to the list
    prompt=(
        "You are an expert research assistant specializing in academic writing. "
        "Your role is to help find credible sources, summarize key findings, and ensure clarity and coherence in research papers. "
        "Use the search_info tool to retrieve relevant information when needed."
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


def create_agent_graph():
    # Initialize the graph
    graph = StateGraph(AgentState)

    # Add the agent node
    graph.add_node("crewai_agent", call_crew_agent)
    graph.add_node("react_agent", call_react_agent)

    # Set the entry point - start with the react_agent
    graph.add_edge(START, "crewai_agent")

    # Connect react_agent to crewai_agent
    graph.add_edge("crewai_agent", "react_agent")

    # Set the exit point - end after the crewai_agent
    graph.add_edge("react_agent", END)

    # Compile the graph
    return graph.compile()


# Create the graph
agent_system = create_agent_graph()

# Example of using the agent system


def run_agent_example():
    # Define a query for our agent
    query = "Tell me about artificial intelligence"

    # Invoke the agent system with the query
    result = agent_system.invoke({
        "messages": [HumanMessage(content=query)]
    })
    print(result)
    # Print the conversation
    print("=== Agent Conversation ===")
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"\nHuman: {message.content}")
            print("-------------------------------------------------")
        elif isinstance(message, AIMessage):
            print(f"\nAI: {message.content}")
            if hasattr(message, "tool_calls") and message.tool_calls:
                print(f"Tool Calls: {message.tool_calls}")
            print("-------------------------------------------------")
        elif isinstance(message, ToolMessage):
            print(f"\nTool ({message.name}): {message.content}")
            print("-------------------------------------------------")

        # Print the final response
    print("\n=== Final Response ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    run_agent_example()
