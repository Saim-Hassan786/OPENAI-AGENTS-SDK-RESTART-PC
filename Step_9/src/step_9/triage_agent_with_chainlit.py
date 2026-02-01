import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from openai.types.responses import ResponseTextDeltaEvent
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    TResponseInputItem
)

external_client = AsyncOpenAI(api_key=GOOGLE_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# Agents setup
french_agent = Agent(
    name="French Agent",
    instructions="You are a French-speaking agent. Respond to all queries in French.",
    model="gemini-2.0-flash",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You are a Spanish-speaking agent. Respond to all queries in Spanish.",
    model="gemini-2.0-flash",
)

english_agent = Agent(
    name="English Agent",
    instructions="You are an English-speaking agent. Respond to all queries in English.",
    model="gemini-2.0-flash",
)

german_agent = Agent(
    name="German Agent",
    instructions="You are a German-speaking agent. Respond to all queries in German.",
    model="gemini-2.0-flash",
)

russian_agent = Agent(
    name="Russian Agent",
    instructions="You are a Russian-speaking agent. Respond to all queries in Russian.",
    model="gemini-2.0-flash",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You are a triage agent. Determine the request or language of the query and always route it to the appropriate language-specific agent.Do not reply by your self unless you detect a language out of your handoffs",
    model="gemini-2.0-flash",
    handoffs=[french_agent, spanish_agent, english_agent, german_agent, russian_agent]
)

# For Chainlit
import chainlit as cl

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Welcome To The Translator Triage Agent").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    # 1️⃣ Ask Triage Agent to decide where to route
    triage_result = Runner.run_streamed(
        triage_agent,
        input=[{"role": "user", "content": message.content}]
    )
    
    async for event in triage_result.stream_events():
        pass  
    
    # Get the agent that should handle the reply
    selected_agent = triage_result.current_agent
    await cl.Message(content=f"Handing off to: {selected_agent.name}").send()

    # 2️⃣ Add latest user message to history
    history.append({"role": "user", "content": message.content})

    # 3️⃣ Ask the selected agent to respond (with full history)
    msg = cl.Message(content="")
    await msg.send()

    agent_result = Runner.run_streamed(
        selected_agent,
        input=history
    )
    
    async for event in agent_result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # 4️⃣ Add assistant response to history
    history.append({"role": "assistant", "content": agent_result.final_output})
    cl.user_session.set("history", history)

    # 5️⃣ Show agent that replied
    await cl.Message(content=f"Replied By: {selected_agent.name}").send()

