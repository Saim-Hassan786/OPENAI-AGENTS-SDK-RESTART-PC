import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import Literal, Any

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    function_tool,
    ToolsToFinalOutputResult,
    FunctionToolResult,
    ModelSettings,
    RunContextWrapper,
    ToolsToFinalOutputFunction,
)

external_client = AsyncOpenAI(
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY")
)
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class Weather(BaseModel):
    city : str
    temperature : float
    condition : str
    humidity : int


@function_tool
def get_weather(city:str)->Weather:
    """"
    Gets the weather for a given city.
    Args:
        city (str): The name of the city to get the weather for.
    Returns:
        Weather: A Weather object containing the weather information for the city.
    """
    return Weather(
        city=city,
        temperature=25.0,
        condition="Sunny",
        humidity=50
    )


async def custom_tool_use(context : RunContextWrapper[Any],results:list[FunctionToolResult])->ToolsToFinalOutputResult:
    weather : Weather = results[0].output
    return ToolsToFinalOutputResult(
        is_final_output=True,
        final_output=f"Weather in {weather.city} is {weather.temperature}°C with {weather.condition} and {weather.humidity}%."
    )


async def main(tool_use_behavior : Literal["default","custom","first_tool_use"]="default"):
    if tool_use_behavior == "default":
        behavior : Literal["run_llm_again","stop_on_first_tool"] | ToolsToFinalOutputFunction = "run_llm_again"
    elif tool_use_behavior == "first_tool_use":
        behavior = "stop_on_first_tool"
    elif tool_use_behavior == "custom":
        behavior = custom_tool_use
    
    agent = Agent(
        name = "Weather Agent",
        instructions="Get the weather for a given city.",
        tools=[get_weather],
        tool_use_behavior=behavior,
        model_settings=ModelSettings(
            tool_choice="required" if tool_use_behavior != "default" else None
        ),
        model="gemini-2.0-flash"
    )
    result = await Runner.run(
        agent,
        "what is the weather in San Francisco?",
    )
    print(result.final_output)

asyncio.run(main("first_tool_use"))
# asyncio.run(main("default"))
# asyncio.run(main("custom"))