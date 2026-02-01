import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Literal, Any

import chainlit as cl

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from agents import (
    Agent,
    Runner,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
    function_tool,
    ToolsToFinalOutputResult,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    ToolsToFinalOutputFunction,
)

# Configure external LLM client
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# Pydantic model for tool output
class Weather(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int


# Define the tool
@function_tool
def get_weather(city: str) -> Weather:
    return Weather(city=city, temperature=25.0, condition="Sunny", humidity=50)


# Custom tool‐use behavior
async def custom_tool_use(
    context: RunContextWrapper[Any],
    results: list[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    weather = results[0].output
    return ToolsToFinalOutputResult(
        is_final_output=True,
        final_output=(
            f"Custom Tool: Weather in {weather.city} is "
            f"{weather.temperature}°C, {weather.condition}, {weather.humidity}% RH."
        ),
    )


# Agent factory
def build_weather_agent(
    mode: Literal["default", "first_tool_use", "custom"] = "default"
) -> Agent:
    if mode == "default":
        behavior: Literal["run_llm_again", "stop_on_first_tool"] = "run_llm_again"
    elif mode == "first_tool_use":
        behavior = "stop_on_first_tool"
    else:
        behavior = custom_tool_use

    return Agent(
        name="Weather Agent",
        instructions="Get the weather for a given city.",
        tools=[get_weather],
        tool_use_behavior=behavior,
        model_settings=ModelSettings(
            tool_choice="required" if mode != "default" else None
        ),
        model="gemini-2.0-flash",
    )


# Initialize a default agent; you can change to "first_tool_use" or "custom"
AGENT = build_weather_agent("custom")


# Chainlit handlers
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="👋 Welcome to the Weather Agent!").send()


@cl.on_message
async def on_message(message: cl.Message):
    # Build or rebuild agent if you want dynamic modes:
    # mode = cl.user_session.get("mode") or "first_tool_use"
    # agent = build_weather_agent(mode)
    agent = AGENT

    # Run the agent using non‑streaming API to honor first_tool_use
    result = await Runner.run(agent, input=message.content)

    # Send final output directly
    await cl.Message(content=result.final_output).send()
