import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    ItemHelpers,
    trace,
    set_trace_processors,
    set_tracing_disabled
)
import asyncio
from openai import AsyncOpenAI
from agents.tracing.processor_interface import TracingProcessor
from pprint import pprint

external_client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
set_default_openai_client(client=external_client)
set_default_openai_api("chat_completions")

# The Code Below Is For Parallelization
spanish_translator_agent = Agent(
    name="Spanish Translator",
    instructions="Translate the given text to Spanish",
    model = "gemini-2.0-flash",
)

best_translation_picker_agent = Agent(
    name="Best Translation Picker",
    instructions="Given a list of translations, pick the best one based on accuracy and fluency.",
    model = "gemini-2.0-flash",
)

set_tracing_disabled(True)  # Disable tracing for parallel execution


# This code below is for chainlit
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

@cl.on_chat_start
async def handle_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Welcome to the Spanish Translation Service! Please enter the text you want to translate."
    ).send()

@cl.on_message
async def main(message:cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()
    result_1,result_2,result_3 = await asyncio.gather(
            Runner.run(spanish_translator_agent,history),
            Runner.run(spanish_translator_agent,history),
            Runner.run(spanish_translator_agent,history)
        )
    results = [
            ItemHelpers.text_message_outputs(result_1.new_items),
            ItemHelpers.text_message_outputs(result_2.new_items),
            ItemHelpers.text_message_outputs(result_3.new_items)
        ]
    translations = "\n\n".join(results)
    i = 0
    for result in results:
        i += 1
        await msg.stream_token(f"\n\nTranslation Result No.{i} :\n\n{result}")
    best_selected_translation =  Runner.run_streamed(
            best_translation_picker_agent,
            input=f"Input: {message.content}\n\nTranslations:\n{translations}"
            )
    async for event in best_selected_translation.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
        history.append({"role":"assistant","content":best_selected_translation.final_output})
        cl.user_session.set("history",history)
    











# async def main():
#     user_input = input("Enter the text to translate: ")
#     # with trace("Translation Tracing"):
#     result_1,result_2,result_3 = await asyncio.gather(
#             Runner.run(spanish_translator_agent,user_input),
#             Runner.run(spanish_translator_agent,user_input),
#             Runner.run(spanish_translator_agent,user_input)
#         )
#     print(f"\n\nTranslation Results:\n\n Runner1\n\n{result_1.final_output}\n\nRunner2\n\n{result_2.final_output}\n\nRunner3\n\n{result_3.final_output}")
#     results = [
#             ItemHelpers.text_message_outputs(result_1.new_items),
#             ItemHelpers.text_message_outputs(result_2.new_items),
#             ItemHelpers.text_message_outputs(result_3.new_items)
#         ]
#     translations = "\n\n".join(results)
#     print(f"\n\nCompiled Translation Results:\n\n{translations}")

#     best_selected_translation = await Runner.run(
#             best_translation_picker_agent,
#             input=f"Input: {user_input}\n\nTranslations:\n{translations}"
#             )
#     print(f"\n\nBest Selected Translation:\n\n{best_selected_translation.final_output}")

# asyncio.run(main())