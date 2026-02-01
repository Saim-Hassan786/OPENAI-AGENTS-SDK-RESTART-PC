import os
import chainlit as cl
from dotenv import load_dotenv
load_dotenv()
from agents import Agent , Runner , AsyncOpenAI , OpenAIChatCompletionsModel
from agents.run import RunConfig
from openai.types.responses import ResponseTextDeltaEvent

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

# Chainlit Integration To Our Agent

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello!I am an AI Agent developed by Saim Hassan. How can I assist you today?").send()

@cl.on_message
async def main(message : cl.Message):
    history = cl.user_session.get("history")
    history.append({"role":"user","content":message.content})
    msg = cl.Message(content="")
    await msg.send()
    result = Runner.run_streamed(
        agent,
        input=history,
        run_config=config,
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    history.append({"role":"assistant","content":result.final_output})
    cl.user_session.set("history",history)        
