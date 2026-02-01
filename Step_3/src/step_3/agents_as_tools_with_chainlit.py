import os
import asyncio
from dotenv import load_dotenv
load_dotenv()


# For Chainlit
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

# For Google Gemini
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

# Now Running The Chainlit
@cl.on_chat_start
async def handle_chat_start():
    await cl.Message(content= "Welcome To The Translator Assistant ! ").send()
    while True:
        ask_input = await cl.AskUserMessage(content="Enter the text you want to be Translated :").send()
        if not ask_input:
           return
        text_to_translate = ask_input["output"]
        ask_lang = await cl.AskUserMessage(content="Enter the languages you want to translate to (Comma Separated) :").send()
        if not ask_lang:
           return
        langs = [l.strip() for l in ask_lang["output"].split(",")]
        result = await Runner.run(
            orchestrator_agent,
            input = f"Translate the following text to {langs}:\n\n{text_to_translate}",
            run_config = config,
    )
        await cl.Message(content=f"**Translation:**\n{result.final_output}").send()
        await cl.Message(content = "Evaluating the Translations").send()
        evaluation_result = await Runner.run(
            evaluator_agent,
                (f"Evaluate only the following translations:\n\n"
                f"{result.final_output}\n\n"
                f"into the languages {langs}. Correct any mistakes; "
                f"reply with 'Perfect : No Changes Needed' or "
                f"'Needed Some Improvements,So Improved' and then the final text."
            ),
            run_config=config,
        )
        await cl.Message(content=f"**Evaluation Result:**\n{evaluation_result.final_output}").send()
        continue_input = await cl.AskUserMessage(content="Do you want to translate another Text ? (yes/no)").send()
        if not continue_input or continue_input["output"].lower() != "yes":
            await cl.Message(content="Thank you for using the Translator Assistant!").send()
            break



