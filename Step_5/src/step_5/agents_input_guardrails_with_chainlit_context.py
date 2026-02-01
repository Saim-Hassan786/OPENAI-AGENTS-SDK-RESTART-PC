import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

# For Chainlit
import chainlit as cl
from openai.types.responses import ResponseTextDeltaEvent

from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    set_default_openai_client,
    set_default_openai_api,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
    InputGuardrail,
    AsyncOpenAI
)


external_client = AsyncOpenAI(
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY"),
)
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class HomeworkMathOutput(BaseModel):
    reasoning: str = Field(description="The reasoning behind the answer.")
    is_math_homework: bool = Field(description="Whether the question is math homework or not")


# Guardrail Agent
guardrail_agent = Agent(
    name = "guardrail_agent",
    instructions = "Check whether the user's question is math homework or not. If it is, provide the reasoning behind your answer.",
    model = "gemini-2.0-flash",
    output_type = HomeworkMathOutput,
)   

@input_guardrail
async def mathhomework_guardrail(context : RunContextWrapper[None], agent : Agent , input : str | list[TResponseInputItem])->GuardrailFunctionOutput:
    """
    This guardrail checks if the input is math homework or not.
    If it is, it provides the reasoning behind the answer.
    """
    result = await Runner.run(
       guardrail_agent,
       input = input,
       context = context.context
   )
    print("TripWire Is Triggered")
    final_output = result.final_output_as(HomeworkMathOutput)
    return GuardrailFunctionOutput(
        output_info = final_output,
        tripwire_triggered = final_output.is_math_homework,
    )

# Setting up our Customer Support Agent
customer_support_agent = Agent(
    name = "customer_support_agent",
    instructions = "You are a customer support agent. Answer the user's question.",
    model = "gemini-2.0-flash",
    input_guardrails = [mathhomework_guardrail]
)

# Now Run With Chainlit
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Welcome to the Customer Support Agent!").send()

@cl.on_message
async def main(message: cl.Message):
    # 1. Retrieve and update the full history from session
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    # 2. Optimistically run the agent (and guardrails) on the entire history
    try:
        result = await Runner.run(
            customer_support_agent,
            input=history
        )
        # 3a. If no tripwire, stream the assistant's reply
        final_result = (result.final_output)
        message = cl.Message(content=final_result)
        await message.send()

        # 4. Refresh history to exactly what the agent saw & allowed
        clean_history = result.to_input_list()
        cl.user_session.set("history", clean_history)

    except InputGuardrailTripwireTriggered:
        # 5a. On tripwire, send refusal
        await cl.Message(
            content="⚠️ Sorry, I can’t help you with your math homework."
        ).send()

        # 5b. Remove the offending user turn so it doesn't block future messages
        history.pop()
        cl.user_session.set("history", history)


