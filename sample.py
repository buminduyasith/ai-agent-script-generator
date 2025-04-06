import os
import sys
import traceback
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import (AIMessage, AIMessageChunk, HumanMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# LangGraph imports
# REMOVE Checkpointer imports for this test
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.errors import GraphInterrupt, NodeInterrupt
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

load_dotenv()

# --- Configuration ---
MODEL_NAME = "gpt-4o-mini"
# DEFAULT_THREAD_ID = "weather_conv_001" # Not needed without checkpointer

# --- Initialize Model ---
model = ChatOpenAI(model=MODEL_NAME, temperature=0)

# --- Define Tool with Interruption ---
@tool
def get_weather(location: str) -> str:
    """
    Use this to get weather information from a given location ONLY AFTER confirming with the user.
    It will pause and ask for confirmation before proceeding.
    """
    print(f"\n[Tool Call]: Attempting to use 'get_weather' for location '{location}'. Requesting confirmation...")
    confirmed = yield Send(
        {"type": "confirm_tool_use", "tool_name": "get_weather", "location": location}
    )
    print(f"[Tool Call]: Received confirmation response: '{confirmed}'")
    if str(confirmed).strip().lower() == 'yes':
        print("[Tool Call]: User confirmed. Proceeding with weather API call simulation...")
        weather_result = f"Okay, fetching weather: It might be cloudy in {location} with a chance of rain today (Sunday, April 6, 2025)."
        print(f"[Tool Call]: Simulated result: {weather_result}")
        return weather_result
    else:
        print("[Tool Call]: User denied permission.")
        return f"Okay, I will not fetch the weather for {location} because permission was denied."

# --- Create Prebuilt Agent ---
agent_executor = create_react_agent(
    model,
    tools=[get_weather],
)

# --- Assign the already compiled agent executor ---
graph = agent_executor

# --- Helper Function ---
def process_messages(messages):
    if not messages: return None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if msg.content: return msg.content
    return None

# --- Main Conversation Loop Function (No Checkpointer) ---
def start_conversation():
    """
    Start a conversation with the agent, handling interruptions (in-memory only).
    """
    # Base configuration - thread_id not relevant without checkpointer
    config = {"configurable": {"thread_id": "in_memory_session"}} # Still need thread_id structure

    print(f"Starting IN-MEMORY conversation (no persistence).")
    print("Enter 'exit' to quit.")

    # Store history manually for this test since checkpointer is removed
    current_messages = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Ending conversation.")
            break
        if not user_input.strip():
            continue

        # Add user message to our manual history
        current_messages.append(HumanMessage(content=user_input))

        # Package the current history for the agent
        inputs = {"messages": current_messages}

        try:
            print("\n[Agent thinking...]")
            # Pass the standard config (no checkpointer)
            # Use invoke directly, as stream adds complexity here
            # We need the final result after potential interruption/resume
            # result = graph.invoke(inputs, config) # Initial attempt

            # Need to handle interruption even without checkpointer
            final_result = None
            while True: # Loop to handle potential resume
                try:
                    # Use stream to allow interruption detection
                    events = graph.stream(inputs, config, stream_mode="values")
                    for event in events:
                        final_result = event # Capture the last state

                    # If stream finished without interruption, break inner loop
                    break

                except (GraphInterrupt, NodeInterrupt) as e:
                    print(f"\n[SYSTEM] INTERACTION REQUIRED (Graph Paused)")
                    if isinstance(e.data, dict) and e.data.get("type") == "confirm_tool_use":
                        tool_name = e.data.get("tool_name")
                        location = e.data.get("location")
                        print(f"   The agent wants to use the tool '{tool_name}' for location '{location}'.")
                        while True:
                           user_response = input(f"   Allow this tool use? (yes/no): ").strip().lower()
                           if user_response in ['yes', 'no']: break
                           else: print("   Please answer 'yes' or 'no'.")

                        try:
                             print("\n[Agent resuming...]")
                             # Resume by invoking again, passing None input and resume value
                             # Still need config for context, even if no checkpointer
                             # The graph state is held in memory by the LangGraph runner
                             resumed_events = graph.stream(None, config, stream_mode="values", resume_from=user_response)
                             for event in resumed_events:
                                 final_result = event # Capture the last state after resume

                             # Break inner loop after successful resume
                             break

                        except Exception as resume_error:
                             print(f"\n[SYSTEM] Error resuming graph: {str(resume_error)}")
                             traceback.print_exc()
                             # Decide how to proceed - break, continue outer, etc.
                             break # Break inner loop on resume error
                    else:
                        print(f"\n[SYSTEM] Unhandled interruption: {e.data}")
                        # Handle other interruptions or break
                        break # Break inner loop on unhandled interruption

            # Process the final result after stream/resume completes
            if final_result and isinstance(final_result, dict) and 'messages' in final_result:
                 # Update our manual history with the final state from the agent
                 current_messages = final_result['messages']
                 ai_response = process_messages(current_messages)
                 if ai_response:
                      print(f"\nAssistant: {ai_response}")
                 else:
                      # Agent might have added tool messages but no final AI message yet
                      # Check the very last message
                      last_msg = current_messages[-1] if current_messages else None
                      if isinstance(last_msg, ToolMessage):
                            print(f"\n[SYSTEM] Tool '{last_msg.name}' executed. Waiting for next step or final response.")
                      else:
                            print("\n[SYSTEM] Agent finished, but no final AI message content found.")

            else:
                 print("\n[SYSTEM] Agent finished, but final state wasn't in the expected format.")


        except Exception as e:
            print(f"\n[SYSTEM] An unexpected error occurred: {str(e)}")
            traceback.print_exc()
            # Reset history on major error? Or try to continue?
            # current_messages = [] # Optional: Reset history on error

# --- Run the Conversation ---
if __name__ == "__main__":
    start_conversation()