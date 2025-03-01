from copy import deepcopy
import os
from crewai import Agent, Task, Crew
import logging
from langchain_core.messages import convert_to_openai_messages, AIMessage
from crewai.tools import tool
model_name = "gpt-4o-mini"
# crewai agents


@tool("search on internet")
def search_on_internet(userId: str) -> str:
    """using this tool you can get users internent package details based on the user id

       Args:
        userId:  users id

    """
    # Function logic here
    data = "debugger"
    return """
    {
        "package":"lite,
        "data_for_month:100GB",
        "price":10000,
        "validity":"30 days",
        "data_speed":"100mbps",
        "data_usage":"100GB",
        "remaining_data":"0GB",
    }
    """


researcher = Agent(
    role="Usage Analyzer",
    goal="you should be able to find user interent package info and usage data based on the user id",
    backstory="once you got user id you will use the necssaery tool and get user usage data. You know when you have enough information to complete your task.",
    verbose=True,
    allow_delegation=False,
    tools=[search_on_internet],
    llm_model=model_name,
    # Add max iterations to prevent infinite loops
    max_iter=3
)

# Define tasks
research_task = Task(
    description="Analyze the usage data of a user based on the user id.",
    agent=researcher,
    expected_output="The user's internet package details and usage data in json format nothing else",
)

# Define crew
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True,
    # Add a specific process to make execution more predictable
    process="sequential"
)


def call_crew_agent(state: any):
    """Node function for the agent"""
    print("Starting call_crew_agent...")
    state_clone = deepcopy(state["messages"])
    messages = convert_to_openai_messages(state["messages"])
    topic = messages[-1]
    print(f"CrewAI topic: {topic}")
    crew_response = "this test msg"
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

    state_clone.append(AIMessage(content=crew_response))
    print(f"CrewAI response (truncated): {crew_response[:100]}...")
    state["messages"] = state_clone
    print("Finished call_crew_agent")
    return {"messages": state["messages"]}



Agent
