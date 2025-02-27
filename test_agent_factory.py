"""
Test script for the Agent Factory

This script demonstrates how to use the agent factory to generate agent code
based on different frameworks.
"""

import json
from agent_factory import generate_agent_code, parse_agent_json


def test_crewai_agent():
    """Test generating a CrewAI agent."""
    json_str = '''
    {
      "agent": {
        "framework": "crewai",
        "role": "Market Research Analyst",
        "systemPrompt": "Provide up-to-date market analysis of the AI industry",
        "expectedOutput": "An expert analyst with a keen eye for market trends",
        "model": "gpt-4",
        "tools": ["search_tool", "web_rag_tool"]
      }
    }
    '''
    
    agent_config = parse_agent_json(json_str)
    agent_code = generate_agent_code(agent_config)
    
    print("=== CrewAI Agent Code ===")
    print(agent_code)
    print("========================")


def test_pydantic_agent():
    """Test generating a Pydantic AI agent."""
    json_str = '''
    {
      "agent": {
        "framework": "pydantic-ai",
        "systemPrompt": "You are a helpful AI assistant",
        "model": "gpt-4o-mini",
        "tools": ["search_wikipedia", "get_current_weather"]
      }
    }
    '''
    
    agent_config = parse_agent_json(json_str)
    agent_code = generate_agent_code(agent_config)
    
    print("=== Pydantic AI Agent Code ===")
    print(agent_code)
    print("=============================")


if __name__ == "__main__":
    test_crewai_agent()
    print("\n")
    test_pydantic_agent()
