import os
import asyncio
import json
from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel, Field

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    GuardrailFunctionOutput,
    RunContextWrapper,
    output_guardrail,
    OutputGuardrailTripwireTriggered,
    AsyncOpenAI
)

# Set external API client
external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY")
)
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

# Output from assistant
class MessageOutput(BaseModel):
    response: str = Field(description="Response in natural language to the user")

# Guardrail check
class OutputCheck(BaseModel):
    is_name: bool = Field(description="Contains name of a person")
    is_phone_num: bool = Field(description="Contains phone number")

# Guardrail agent
guardrail_instructions = """
You will be given a piece of assistant-generated text.
Output a JSON with:
  - is_user_name: true if it includes a person’s name, false otherwise
  - is_phone_num: true if it includes a phone number, false otherwise

Respond only in JSON.

Example:
Text: "Hi Alice, my number is 123-456-7890"
Output: {"is_user_name": true, "is_phone_num": true}

Now analyze this:
Text: "{ASSISTANT_RESPONSE_HERE}"
"""

guardrail_output_agent = Agent(
    name="Guardrail Check",
    instructions=guardrail_instructions,
    output_type=OutputCheck,
    model="gemini-2.0-flash"
)

# Guardrail function
@output_guardrail
async def output_guardrail_check(context: RunContextWrapper, agent: Agent, output: MessageOutput) -> GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_output_agent,
        output.response,
        context=context.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_name or result.final_output.is_phone_num
    )

# Main agent
agent = Agent(
    name="Customer Assistant",
    instructions="You are a helpful assistant. ONLY respond in raw text like: Hello! How can I help you today?",
    model="gemini-2.0-flash",
    output_type=MessageOutput,
    output_guardrails=[output_guardrail_check]
)

# Chainlit start
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to the Customer Assistant!").send()

# Main handler
@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    
    try:
        # First run without streaming to check for guardrails
        validation_result = await Runner.run(
            agent,
            input=history
        )
        
        # If validation passed without triggering guardrails, then stream the response
        msg = cl.Message(content="")
        await msg.send()
        
        buffer = ""
        result = Runner.run_streamed(
            agent,
            input=history
        )
        
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                buffer += event.data.delta
                try:
                    parsed = json.loads(buffer)
                    display_text = parsed["response"]
                    await msg.stream_token(display_text)
                    buffer = ""
                except json.JSONDecodeError:
                    pass
        
        history.append({"role": "assistant", "content": validation_result.final_output.response})
        
    except OutputGuardrailTripwireTriggered:
        await cl.Message(
            content="⚠️ Sorry, I can't share this response as it contains sensitive information"
        ).send()
        history.append({"role": "assistant", "content": "Response blocked due to sensitive content"})
    
    cl.user_session.set("history", history)
