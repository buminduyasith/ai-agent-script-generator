"""
Agent Factory Module

This module implements a factory design pattern for creating AI agents based on
different frameworks (crewAI, Pydantic AI, etc.) from user input specifications.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json


class AgentFactory(ABC):
    """Abstract factory for creating agents based on different frameworks."""

    @abstractmethod
    def create_agent(self, agent_config: Dict[str, Any]) -> str:
        """
        Create agent code based on the configuration.

        Args:
            agent_config: Dictionary containing agent configuration

        Returns:
            String representation of the generated agent code
        """
        pass


class CrewAIAgentFactory(AgentFactory):
    """Factory for creating CrewAI agents."""

    def create_agent(self, agent_config: Dict[str, Any]) -> str:
        """Create a CrewAI agent based on the configuration."""
        role = agent_config.get("role", "")
        system_prompt = agent_config.get("systemPrompt", "")
        expected_output = agent_config.get("expectedOutput", "")
        model = agent_config.get("model", "gpt-4")
        tools = agent_config.get("tools", [])

        # Here we would generate the actual code for a CrewAI agent
        # For now, we'll just return a template string

        tools_str = ", ".join([f'"{tool}"' for tool in tools])

        code = f'''
from crewai import Agent

# Create a CrewAI agent
agent = Agent(
    role="{role}",
    goal="{expected_output}",
    backstory="{system_prompt}",
    verbose=True,
    allow_delegation=False,
    llm="{model}",
    tools=[{tools_str}]
)
'''
        return code


class PydanticAIAgentFactory(AgentFactory):
    """Factory for creating Pydantic AI agents."""

    def create_agent(self, agent_config: Dict[str, Any]) -> str:
        """Create a Pydantic AI agent based on the configuration."""
        system_prompt = agent_config.get("systemPrompt", "")
        model = agent_config.get("model", "gpt-4")
        tools = agent_config.get("tools", [])

        # Convert model name to Pydantic AI format if needed
        if ":" not in model:
            model = f"openai:{model}"

        # Here we would generate the actual code for a Pydantic AI agent
        # For now, we'll just return a template string

        tools_code = ""
        for tool in tools:
            tools_code += f'''
@agent.tool
def {tool}(query: str) -> str:
    """
    {tool.replace('_', ' ').title()}
    
    Args:
        query: The query to process
        
    Returns:
        The result of the {tool} operation
    """
    # Implementation for {tool}
    return f"Result for {{query}} using {tool}"
'''

        code = f'''
from pydantic_ai import Agent
from pydantic_ai.run import RunContext
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create a Pydantic AI agent
agent = Agent(
    "{model}",
    system_prompt="{system_prompt}"
)

{tools_code}

# Example usage
def run_agent(query):
    try:
        result = agent.run_sync(query)
        return result.data
    except Exception as e:
        return f"Error: {{e}}"
'''
        return code


class AgentFactoryProvider:
    """Provider class to get the appropriate factory based on the framework."""

    @staticmethod
    def get_factory(framework: str) -> AgentFactory:
        """
        Get the appropriate factory based on the framework.

        Args:
            framework: The name of the framework (e.g., 'crewai', 'pydantic-ai')

        Returns:
            An instance of the appropriate AgentFactory

        Raises:
            ValueError: If the framework is not supported
        """
        framework = framework.lower()

        if framework == "crewai":
            return CrewAIAgentFactory()
        elif framework in ["pydantic-ai", "pydantic_ai", "pydanticai"]:
            return PydanticAIAgentFactory()
        else:
            raise ValueError(f"Unsupported framework: {framework}")


def generate_agent_code(agent_config: Dict[str, Any]) -> str:
    """
    Generate agent code based on the configuration.

    Args:
        agent_config: Dictionary containing agent configuration

    Returns:
        String representation of the generated agent code

    Raises:
        ValueError: If the framework is not specified or not supported
    """
    framework = agent_config.get("framework")
    if not framework:
        raise ValueError("Framework not specified in agent configuration")

    factory = AgentFactoryProvider.get_factory(framework)
    return factory.create_agent(agent_config)


def parse_agent_json(json_str: str) -> Dict[str, Any]:
    """
    Parse JSON string to extract agent configuration.

    Args:
        json_str: JSON string containing agent configuration

    Returns:
        Dictionary containing agent configuration
    """
    try:
        data = json.loads(json_str)
        return data.get("agent", {})
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string")


# Example usage
if __name__ == "__main__":
    # Example JSON input
    example_json = '''
    {
      "agent": {
        "framework": "crewai",
        "role": "Market Research Analyst",
        "systemPrompt": "Provide up-to-date market analysis of the AI industry",
        "expectedOutput": "An expert analyst with a keen eye for market trends",
        "model": "gpt-4",
        "tools": ["web_api_tool", "rag_tool"]
      }
    }
    '''

    # Parse JSON
    agent_config = parse_agent_json(example_json)

    # Generate agent code
    try:
        agent_code = generate_agent_code(agent_config)
        print(agent_code)
    except ValueError as e:
        print(f"Error: {e}")
