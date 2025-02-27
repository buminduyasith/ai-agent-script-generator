import autogen
import os
from langchain_core.messages import convert_to_openai_messages

model_name = "gpt-4o-mini"

config_list = [{"model": model_name, "api_key": os.environ["OPENAI_API_KEY"]}]

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list,
    "temperature": 0,
}

autogen_agent = autogen.AssistantAgent(
    name="RecommendationGenerator",
    llm_config=llm_config,
    system_message="""
    I am a recommendation generator. Please provide me with the information I need to generate a recommendation.
    you will receive a user internet package details and usage data based on the user id. based on that you have to generate a recommendation.
    Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.
    """,
)


def test(query: str) -> str:
    return "test"


user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get(
        "content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "web",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    llm_config=llm_config,
    system_message="Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.",
)


# result = autogen_agent.initiate_chat(
#     user_proxy, message="""
#        {
#         "package":"lite,
#         "data_for_month:100GB",
#         "price":10000,
#         "validity":"30 days",
#         "data_speed":"100mbps",
#         "data_usage":"100GB",
#         "remaining_data":"0GB",
#     }
#     """)
# print(result.chat_history[-1]["content"])


def call_autogen_agent(state: any):
    # convert to openai-style messages
    print("Starting call_autogen_agent...")
    messages = convert_to_openai_messages(state["messages"])
    response = user_proxy.initiate_chat(
        autogen_agent,
        message=messages[-1],
        # pass previous message history as context
        carryover=messages[:-1],
    )
    # get the final response from the agent
    content = response.chat_history[-1]["content"]
    return {"messages": {"role": "assistant", "content": content}}
