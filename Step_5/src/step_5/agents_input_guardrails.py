import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field

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

# Now Running The Loop
async def main():
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            result = await Runner.run(
                customer_support_agent,
                input = user_input,
            )
            print(result.final_output)
        except InputGuardrailTripwireTriggered as e:
            print("Tripwire triggered!")
            print("The Following Question is Not Allowed, we cannot help you with Math-Related Questions")

asyncio.run(main())