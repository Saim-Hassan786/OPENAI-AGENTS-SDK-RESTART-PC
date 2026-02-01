import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

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
    OutputGuardrailTripwireTriggered
)
from agents import AsyncOpenAI

external_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY")
)
set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

class MessageOutput(BaseModel):
    response : str = Field(description="Response in relative to the user's query")

class OutputCheck(BaseModel):
    is_name : bool = Field(description="Containing name of a person")
    is_phone_num : bool = Field(description="Containing phone number of a person") 


guardrail_instructions = """ You will be given a piece of assistant-generated text.  
Your job is to output a JSON object with two fields:
  • is_user_name: true if the text mentions a person's name, otherwise false  
  • is_phone_num: true if the text mentions a phone number in typical formats (e.g. “555-123-4567”, “(123) 456 7890” or any other numerical format that resembles to be a phone number), otherwise false  

Here are some examples:

Example 1:
Text: "Hello John Doe, welcome back!"
Output: {"is_user_name": true, "is_phone_num": false}

Example 2:
Text: "Sure — you can reach me at 555-123-4567 for more details."
Output: {"is_user_name": false, "is_phone_num": true}

Example 3:
Text: "Hi Alice, my number is (123) 456-7890; let me know when you're free."
Output: {"is_user_name": true, "is_phone_num": true}

Now, analyze the following and output only the JSON:
Text: "{ASSISTANT_RESPONSE_HERE}"
"""

guardrail_output_agent = Agent(
    name="Guardrail check",
    instructions=guardrail_instructions,
    output_type=OutputCheck,
    model = "gemini-2.0-flash"
)

@output_guardrail
async def output_guardrail_check(context:RunContextWrapper,agent:Agent,output:MessageOutput)->GuardrailFunctionOutput:
    result = await Runner.run(
        guardrail_output_agent,
        output.response,
        context= context.context
    )
    return GuardrailFunctionOutput(
        output_info= result.final_output,
        tripwire_triggered= result.final_output.is_name or result.final_output.is_phone_num
    )

agent = Agent(
    name = "Customer Assistant",
    instructions="You are a helpful assistant that reply user queries",
    model = "gemini-2.0-flash",
    output_type=MessageOutput,
    output_guardrails=[output_guardrail_check]
)

async def main():
    while True:
        user_input = input("Enter your question (or 'exit' to quit): ")  
        if user_input.lower() == "exit":
            break
        try :
            result = await Runner.run(
                agent,
                input=user_input
            )   
            print(result.final_output.response)
        except OutputGuardrailTripwireTriggered as e :
            print("Tripwire triggered!")
            print("The Following Question contains sensitive information")


asyncio.run(main())

