import autogen
import os
import json
from langchain_core.messages import convert_to_openai_messages


def load_config(config_path="config/agent_config.json"):
    """Load the agent configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_agents(config):
    """Set up the autogen agents based on the configuration."""
    # Extract configuration values
    model_config = config["model_config"]
    agent_config = config["agent_config"]
    user_proxy_config = config["user_proxy_config"]

    # Set up LLM configuration
    config_list = [{"model": model_config["name"],
        "api_key": os.environ["OPENAI_API_KEY"]}]

    llm_config = {
        "timeout": model_config["timeout"],
        "cache_seed": model_config["cache_seed"],
        "config_list": config_list,
        "temperature": model_config["temperature"],
    }

    # Create the recommendation agent
    autogen_agent = autogen.AssistantAgent(
        name=agent_config["name"],
        llm_config=llm_config,
        system_message=agent_config["system_message"],
    )

    # Create the user proxy agent
    user_proxy = autogen.UserProxyAgent(
        name=user_proxy_config["name"],
        human_input_mode=user_proxy_config["human_input_mode"],
        max_consecutive_
