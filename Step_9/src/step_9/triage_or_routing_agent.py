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
    instructions="You are a triage agent. Determine the request or language of the query and always route it to the appropriate language-specific agent.",
    model="gemini-2.0-flash",
    handoffs=[french_agent, spanish_agent, english_agent, german_agent, russian_agent]
)

async def main():
    agent = triage_agent

    while True:
        user_input = input("Enter your query (or type 'exit' to quit'): ")

        if not user_input or user_input.lower() == "exit":
            print("Exiting the triage agent.")
            break

        msg: list[TResponseInputItem] = [{"role": "user", "content": user_input}]
        result = Runner.run_streamed(
            agent,
            input=msg,
        )
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
        print(f"🔄 Using agent: {result.current_agent.name}")
        print("\n")

asyncio.run(main())

