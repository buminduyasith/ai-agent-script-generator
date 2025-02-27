import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Load environment variables from .env file (if you have one)
load_dotenv()

def main():
    """
    Create and run a Pydantic AI agent with custom tools using the gpt-4o-mini model.
    """
    # Create a new agent with gpt-4o-mini model
    agent = Agent(
        'openai:gpt-4o-mini',  # Using the gpt-4o-mini model
        system_prompt="""You are a helpful assistant with access to various tools.
        Use these tools to provide the most accurate and helpful responses to user queries.
        When using tools, make sure to interpret their results correctly.""",
    )

    # Define tools using the decorator pattern

    @agent.tool
    def calculate(ctx: RunContext[None], expression: str) -> Dict[str, Any]:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: The mathematical expression to evaluate (e.g., "2 + 2").
        
        Returns:
            A dictionary containing the result of the calculation.
        """
        try:
            # Using eval is generally not recommended for user input in production
            # This is just for demonstration purposes
            result = eval(expression, {"__builtins__": {}})
            return {
                "expression": expression,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "expression": expression,
                "result": None,
                "error": str(e)
            }

    @agent.tool
    def get_current_weather(ctx: RunContext[None], location: str) -> Dict[str, Any]:
        """
        Get the current weather for a location.
        
        Args:
            location: The name of the city or location to get weather for.
        
        Returns:
            A dictionary containing weather information.
        """
        # In a real implementation, you would use a weather API
        # This is a simplified mock implementation
        return {
            "location": location,
            "temperature": 22,  # Celsius
            "conditions": "Partly cloudy",
            "humidity": 65,  # Percentage
            "wind_speed": 10,  # km/h
            "note": "This is simulated weather data for demonstration purposes."
        }

    @agent.tool
    def search_wikipedia(ctx: RunContext[None], query: str) -> Dict[str, Any]:
        """
        Search for information on Wikipedia.
        
        Args:
            query: The search query to look up on Wikipedia.
        
        Returns:
            A dictionary containing search results.
        """
        # In a real implementation, you would use a Wikipedia API
        # This is a simplified mock implementation
        return {
            "query": query,
            "results": [
                {
                    "title": f"Wikipedia article about {query}",
                    "summary": f"This is a simulated Wikipedia summary about {query}. "
                              f"In a real implementation, this would contain actual information from Wikipedia."
                }
            ]
        }

    # Use a hardcoded question instead of asking for user input
    user_query = "What's the weather in London and calculate 25 * 4?"
    print(f"Question: {user_query}")

    # Run the agent with the user input
    result = agent.run_sync(user_query)

    # Print the result
    print("\nAgent Response:")
    print(result.data)


if __name__ == "__main__":
    # Set a dummy API key for demonstration purposes
    # In a real application, you should use your actual OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
        print("Note: Using a placeholder API key. Replace with your actual OpenAI API key.")
    
    main()
