import os
from dotenv import load_dotenv
from pydantic_ai import Agent

# Load environment variables from .env file (if you have one)
load_dotenv()

def main():
    """
    A simple example of using Pydantic AI with the gpt-4o-mini model.
    """
    # Set a dummy API key for demonstration
    # Replace with your actual OpenAI API key in a real application
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
        print("Note: Using a placeholder API key. Replace with your actual OpenAI API key.")

    # Create a new agent with gpt-4o-mini model
    agent = Agent(
        'openai:gpt-4o-mini',  # Using the gpt-4o-mini model
        system_prompt="You are a helpful assistant that provides clear and concise information.",
    )

    # Define a simple question
    question = "What are the three primary colors?"
    print(f"Question: {question}")

    # Run the agent with the question
    result = agent.run_sync(question)

    # Print the result
    print("\nAgent Response:")
    print(result.data)

    # Print token usage information if available
    try:
        print("\nToken Usage:")
        print(f"Total tokens: {result.usage.total_tokens}")
    except (AttributeError, TypeError):
        print("\nToken usage information not available")

if __name__ == "__main__":
    main()
