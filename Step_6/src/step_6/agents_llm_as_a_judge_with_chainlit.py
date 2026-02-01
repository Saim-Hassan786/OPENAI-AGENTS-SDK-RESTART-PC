import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from dataclasses import dataclass

from agents import Agent,Runner,ItemHelpers,set_tracing_disabled, set_default_openai_api, set_default_openai_client,TResponseInputItem
from openai import AsyncOpenAI

# For Chainlit
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

external_client = AsyncOpenAI(
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key = os.getenv("GOOGLE_API_KEY"),
)

set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

story_outline_generator = Agent(
    name ="Story Outline Generator",
    instructions = "You are a story outline generator. You will be given a story idea and you will generate a story outline of not more than 100 words for it.If there is any feedback given to you by other agent you improve your output as well",
    model = "gemini-2.0-flash"
)

@dataclass
class EvaluationFeedback():
    feedback : str
    score : Literal["pass","needs_improvement","fail"]

evaluator_agent = Agent(
    name = "Evaluator Agent",
    instructions = "You are an evaluator agent. You will be given a story outline and you will evaluate it and gives feedback whether it needs to be improved or not and also gives score one out of these 'pass','needs_improvement, 'fails' ",
    model = "gemini-2.0-flash",
    output_type = EvaluationFeedback,
)

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="🖐 Welcome to Story Outliner!").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    outline_msg = cl.Message(content="✍️ Generating outline…")
    await outline_msg.send()

    outline_text = ""
    outline_result = Runner.run_streamed(story_outline_generator, history)
    async for ev in outline_result.stream_events():
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            outline_text += ev.data.delta
            await outline_msg.stream_token(ev.data.delta)

    history.append({"role": "assistant", "content": outline_text})

    eval_msg = cl.Message(content="🔍 Evaluating outline…")
    await eval_msg.send()

    eval_text = ""
    eval_result = Runner.run_streamed(evaluator_agent, history)
    async for ev in eval_result.stream_events():
        if ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
            eval_text += ev.data.delta
            
    feedback: EvaluationFeedback = eval_result.final_output
    formatted_response = (
        f"🔍 Evaluation Result\n\n"
        f"• Score: {feedback.score}\n"
        f"• Feedback: {feedback.feedback}"
    )
    eval_msg = cl.Message(content="")
    await eval_msg.send()

    CHUNK_SIZE = 30  
    for i in range(0, len(formatted_response), CHUNK_SIZE):
        chunk = formatted_response[i : i + CHUNK_SIZE]
        await eval_msg.stream_token(chunk)

    if feedback.score != "pass":
        history.append({
            "role": "assistant",
            "content": f"Feedback: {feedback.feedback} (Score: {feedback.score})"
        })
        await cl.Message(content="🔄 Let me improve that...").send()
        cl.user_session.set("history", history)
        return await main(message)

    await cl.Message(content=f"✅ Approved!\n\n**Outline:**\n{outline_text}").send()
    cl.user_session.set("history", history)

