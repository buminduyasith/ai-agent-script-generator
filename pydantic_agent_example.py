import os
from dotenv import load_dotenv
from pydantic_ai import Agent

# Load environment variables from .env file (if you have one)
load_dotenv()

def main():
    """
    Create and run a simple Pydantic AI agent using the gpt-4o-mini model.
    """
    # Create a new agent with gpt-4o-mini model
    agent = Agent(
        'openai:gpt-4o-mini',  # Using the gpt-4o-mini model
        system_prompt="You are a helpful assistant that provides clear and concise information.",
    )

    # Get user input
    user_query = input("Enter your question: ")

    # Run the agent with the user input
    result = agent.run_sync(user_query)

    # Print the result
    print("\nAgent Response:")
    print(result.data)

    # You can also access usage information
    print("\nToken Usage:")
    print(f"Input tokens: {result.usage.input_tokens}")
    print(f"Output tokens: {result.usage.output_tokens}")
    print(f"Total tokens: {result.usage.total_tokens}")


if __name__ == "__main__":
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key as an environment variable or in a .env file.")
        api_key = input("Or enter your OpenAI API key here: ")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("No API key provided. Exiting.")
            exit(1)
    
    main()
