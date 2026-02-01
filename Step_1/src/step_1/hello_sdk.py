import os
from dotenv import load_dotenv
load_dotenv()
from agents import Agent , Runner , AsyncOpenAI , OpenAIChatCompletionsModel
from agents.run import RunConfig


external_client = AsyncOpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

agent = Agent(
    name = "Greeting_Model",
    instructions = "A model that greets the user in the same language as the user speaks.",
    model = model,
)

runner = Runner.run_sync(
    agent,
    "ki haal hai tuwada",
    run_config = config,
)

print("Printing Result")
print(runner.final_output)


     
