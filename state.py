from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# 1. Define the State class - this is how messages are managed


class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Create initial state
initial_state: State = {
    "messages": []  # Start with empty messages list
}

# 3. Example: Add a human message


def add_human_message(state: State) -> dict:
    # Create a new human message
    human_msg = HumanMessage(content="Hello, how can you help me?")

    # Return dict with new messages to be added
    return {
        "messages": [human_msg]
    }

# 4. Example: Add an AI response


def add_ai_response(state: State) -> dict:
    # Create an AI response message
    ai_msg = AIMessage(
        content="I'm here to help! What would you like to know?")

    # Return dict with new messages to be added
    return {
        "messages": [ai_msg]
    }

# 5. Example: Read messages from state


def read_messages(state: State):
    """Print all messages in the state"""
    for msg in state["messages"]:
        # Messages have content and type (human/ai)
        print(f"Type: {type(msg).__name__}")
        print(f"Content: {msg.content}")
        print("---")

# Example usage in a workflow:


def example_workflow():
    # Start with empty state
    state = initial_state
    copy_state = state

    # Add human message
    human_update = add_human_message(state)
    # Messages are automatically merged using add_messages annotation
    state["messages"] = add_messages(
        state["messages"], human_update["messages"])

    # Add AI response
    ai_update = add_ai_response(state)
    # New messages are merged with existing ones
    state["messages"] = add_messages(state["messages"], ai_update["messages"])

    # Read all messages
    print("Final state messages:")
    read_messages(state)


if __name__ == "__main__":
    example_workflow()
