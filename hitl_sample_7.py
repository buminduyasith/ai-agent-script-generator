import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define a simple weather tool
@tool
def get_weather(location: str) -> str:
    """Get the weather for a specific location."""
    print(f"Getting weather for {location}...")
    return f"It's currently cloudy in {location} with a chance of rain."

# Define the tools list
tools = [get_weather]

# Create a prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. If asked about weather, you should use the get_weather tool.
    Be friendly and concise in your responses."""),
    MessagesPlaceholder(variable_name="messages"),
])

def start_conversation():
    """Start a conversation with the agent with human-in-the-loop for tool calls."""
    print("Starting conversation. Type 'exit' to quit.")
    
    # Initialize conversation history
    messages = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        # Add user message to history
        messages.append(HumanMessage(content=user_input))
        
        # Get response from the model
        print("\nProcessing...")
        
        # Create the full message list to send to the model
        chain = prompt | model.bind(tools=tools)
        response = chain.invoke({"messages": messages})
        
        # Check if the response contains tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Extract tool call information
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            
            # For weather tool, get the location
            if tool_name == "get_weather" and "location" in tool_args:
                location = tool_args["location"]
                
                # Ask for user confirmation
                print(f"\nThe assistant wants to use the '{tool_name}' tool for location '{location}'")
                user_confirmation = input("Do you want to allow this? (yes/no): ")
                
                if user_confirmation.lower() == "yes":
                    # Execute the tool
                    tool_result = get_weather(location)
                    
                    # Create a tool message
                    tool_message = ToolMessage(
                        content=tool_result,
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                    
                    # Add the response and tool message to history
                    messages.append(response)
                    messages.append(tool_message)
                    
                    # Get final response after tool execution
                    print(f"Tool executed: {tool_result}")
                    final_response = chain.invoke({"messages": messages})
                    
                    # Add final response to history
                    messages.append(final_response)
                    
                    # Print the assistant's response
                    print(f"Assistant: {final_response.content}")
                else:
                    # If user denies, create a tool message with an error
                    tool_message = ToolMessage(
                        content="Tool use was denied by the user.",
                        name=tool_name,
                        tool_call_id=tool_call_id
                    )
                    
                    # Add the response and tool message to history
                    messages.append(response)
                    messages.append(tool_message)
                    
                    # Get final response after tool denial
                    print("Tool use denied. Continuing without using the tool.")
                    final_response = chain.invoke({"messages": messages})
                    
                    # Add final response to history
                    messages.append(final_response)
                    
                    # Print the assistant's response
                    print(f"Assistant: {final_response.content}")
            else:
                # Handle unknown tool
                print(f"Unknown tool requested: {tool_name}")
                messages.append(response)
                print(f"Assistant: {response.content}")
        else:
            # No tool calls, just add the response to history and print it
            messages.append(response)
            print(f"Assistant: {response.content}")

if __name__ == "__main__":
    start_conversation()
