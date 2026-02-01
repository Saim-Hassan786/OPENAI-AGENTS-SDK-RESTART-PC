import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from agents import Agent, Runner, AsyncOpenAI , OpenAIChatCompletionsModel
from agents.run import RunConfig

# Basic Setup
external_client = AsyncOpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client,
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True,
)

# Now Making Our Agents
french_translator_agent = Agent(
    name = "French Translator",
    instructions = "You are a very good French translator.You only translates user's given text to French.",
    model = model,
    handoff_description = "Translate the user's given text to French.",
)

spanish_translator_agent = Agent(
    name = "Spanish Translator",
    instructions = "You are a very good Spanish translator.You only translates user's given text into Spanish.",
    model = model,
    handoff_description = "Translate the user's given text to Spanish.",
)

italian_translator_agent = Agent(
    name = "Italian Translator",
    instructions = "You are a very good Italian translator.You only translates user's given text into Italian.",
    model = model,
    handoff_description = "Translate the user's given text to Italian.",
)

german_translator_agent = Agent(
    name = "German Translator",
    instructions = "You are a very good German translator. You only translates user's given text into German.",
    model = model,
    handoff_description = "Translate the user's given text to German.",
)

russian_translator_agent = Agent(
    name = "Russian Translator",
    instructions = "You are a very good Russian translator. You only translates user's given text into Russian.",
    model = model,
    handoff_description = "Translate the user's given text to Russian.",
)

hindi_translator_agent = Agent(
    name = "Hindi Translator",
    instructions = "You are a very good Hindi translator.  You only translates user's given text into Hindi.",
    model = model,
    handoff_description = "Translate the user's given text to Hindi.",
)

urdu_translator_agent = Agent(
    name = "Urdu Translator",
    instructions = "You are a very good Urdu translator. You only translates user's given text into Urdu.",
    model = model,
    handoff_description = "Translate the user's given text to Urdu.",
)

# Now Making Main Orchestrator Agent
orchestrator_agent = Agent(
    name = "Orchestrator Agent",
    instructions = "You are a translation agent. Your job is to use the tools provided to perform the translations.When asked for multiple translations, you must call each relevant tool in the correct order.You are not allowed to translate anything yourself — always rely on the tools provided",
    model = model,
    tools = [
        french_translator_agent.as_tool(
            tool_name = "french_translator",
            tool_description = "Translate the text to French.",
        ),
        spanish_translator_agent.as_tool(
            tool_name = "spanish_translator",
            tool_description = "Translate the text to Spanish.",
        ),
        italian_translator_agent.as_tool(
            tool_name = "italian_translator",
            tool_description = "Translate the text to Italian.",
        ),
        german_translator_agent.as_tool(
            tool_name = "german_translator",
            tool_description = "Translate the text to German.",
        ),
        russian_translator_agent.as_tool(
            tool_name = "russian_translator",
            tool_description = "Translate the text to Russian.",
        ),
        hindi_translator_agent.as_tool(
            tool_name = "hindi_translator",
            tool_description = "Translate the text to Hindi.",
        ),
        urdu_translator_agent.as_tool(
            tool_name = "urdu_translator",
            tool_description = "Translate the text to Urdu.",
        )
    ]
)

# Now Make A Evaluator Agent
evaluator_agent = Agent(
    name = "Evaluator Agent",
    instructions = "You inspect translations, correct them if needed, and produce a final concatenated response.",
    model = model
)

async def main():
    print("Welcome To Translator AI\n\n")
    input_1 = input("Welcome To Translator AI \n\nEnter Your Text Here Which You Want to be Translated\n\n")
    input_2 = input("\n\nEnter the languages you want to translate to (comma separated):\n\n")
    input_list = [lang.strip() for lang in input_2.split(",")]
    print("\n\nNow Translating The Text\n\n")
    result = await Runner.run(
        orchestrator_agent,
        f"Translate the following text to {input_list}:\n\n{input_1}",
        run_config = config,
    )
    print("Translation Result:\n\n", result.final_output)
    print("\n\nNow Evaluating The Translations\n\n")
    evaluation_result = await Runner.run(
        evaluator_agent,
        f"\n\nEvaluate only the following translations:\n\n{result.final_output} done by previous agent into the given languages {input_list}.\n\nMake sure to correct any mistakes, that's it do not given any other information or explanation just tell in either of sentences 'Perfect : No Changes Needed' or 'Needed Some Improvements,So Improved' and then give your final output'\n\n",
        run_config = config,
    )
    print("Evaluation Result:\n\n", evaluation_result.final_output)

# Now Running The Program
asyncio.run(main())