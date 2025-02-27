import asyncio
from autogen import AssistantAgent, UserProxyAgent
import os

model_name = "gpt-4o-mini"

config_list = [{"model": model_name, "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

assistant_agent = AssistantAgent(
    name="RecommendationGenerator",
    llm_config=llm_config,
    system_message=(
        "I am a recommendation generator. Please provide me with the information I need to generate a recommendation. "
        "You will receive user internet package details and usage data based on the user ID. Based on that, you have to generate a recommendation."
    )
)

user_proxy_agent = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",  # Adjust this based on whether you want human input
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    }
)

user_proxy_agent.initiate_chat(
    assistant_agent,
    message="hi"
)
