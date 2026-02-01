
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
    instructions = "You are a translation agent. Your job is to use the tools provided to perform the translations.When asked for multiple translations, you must call each relevant tool in the correct order.You are not allowed to translate anything yourself — always rely on the tools provided tools.",
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

@cl.on_chat_start
async def init():
    # Initialize state: we're waiting for the text
    cl.user_session.set("step", "awaiting_text")

@cl.on_message
async def main(message: cl.Message):
    step = cl.user_session.get("step")

    if step == "awaiting_text":
        cl.user_session.set("text", message.content)
        cl.user_session.set("step", "awaiting_langs")
        await cl.Message(content="Great! Now, enter target languages (comma separated):").send()
        return

    if step == "awaiting_langs":
        text_to_translate = cl.user_session.get("text")
        langs = [l.strip() for l in message.content.split(",")]

        await cl.Message(content="Translating…").send()

        result = await Runner.run(
            orchestrator_agent,
            f"Translate the following text to {langs}:\n\n{text_to_translate}",
            run_config=config,
        )
        await cl.Message(content=f"**Translation:**\n{result.final_output}").send()

        eval_res = await Runner.run(
            evaluator_agent,
            f"Evaluate only the following translations:\n\n{result.final_output} "
            f"into the languages {langs}. Correct any mistakes; reply with "
            f"'Perfect : No Changes Needed' or 'Needed Some Improvements,So Improved' "
            f"and then the final text.",
            run_config=config,
        )
        await cl.Message(content=f"**Evaluation:**\n{eval_res.final_output}").send()

        cl.user_session.set("step", "awaiting_text")
        return

    await cl.Message(content="Oops, I wasn't ready for that. Let's start over.").send()
    cl.user_session.set("step", "awaiting_text")
