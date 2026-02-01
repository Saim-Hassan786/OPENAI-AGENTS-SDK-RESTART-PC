import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

# For Chainlit
import chainlit as cl

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
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
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
    name="guardrail_agent",
    instructions="Check whether the user's question is math homework or not. If it is, provide the reasoning behind your answer.",
    model="gemini-2.0-flash",
    output_type=HomeworkMathOutput,
)

@input_guardrail
async def mathhomework_guardrail(context: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    """
    This guardrail checks if the input is math homework or not.
    If it is, it provides the reasoning behind the answer.
    """
    result = await Runner.run(
        guardrail_agent,
        input=input,
        context=context.context
    )
    final_output = result.final_output_as(HomeworkMathOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=final_output.is_math_homework,
    )

# Setting up our Customer Support Agent
customer_support_agent = Agent(
    name="customer_support_agent",
    instructions="You are a customer support agent. Answer the user's question.",
    model="gemini-2.0-flash",
    input_guardrails=[mathhomework_guardrail]
)

# Now Run With Chainlit
@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Welcome to the Customer Support Agent!").send()

@cl.on_message
async def main(message : cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})
    msg = cl.Message(content="")
    await msg.send()
    user_input = message.content
    input_data: list[TResponseInputItem] = []
    input_data.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

    try:
            result = await Runner.run(customer_support_agent, input=input_data)
            final_result = (result.final_output)
            message = cl.Message(content=final_result)
            await message.send()
            # If the guardrail didn't trigger, we use the result as the input for the next run
            input_data = result.to_input_list()
    except InputGuardrailTripwireTriggered:
            # If the guardrail triggered, we instead add a refusal message to the input
            message = cl.Message(content="Sorry, I can't help you with your math homework.")
            await message.send()
            input_data.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )