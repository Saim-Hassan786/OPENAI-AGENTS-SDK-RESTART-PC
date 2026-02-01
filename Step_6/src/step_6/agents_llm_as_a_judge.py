import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from dataclasses import dataclass

from agents import Agent,Runner,ItemHelpers,set_tracing_disabled, set_default_openai_api, set_default_openai_client,TResponseInputItem
from openai import AsyncOpenAI

external_client = AsyncOpenAI(
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key = os.getenv("GOOGLE_API_KEY"),
)

set_default_openai_client(external_client)
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

story_outline_generator = Agent(
    name ="Story Outline Generator",
    instructions = "You are a story outline generator. You will be given a story idea and you will generate a story outline of not more than 100 words for it.If there is any feedback given to you by other agent you improve your output as well",
    model = "gemini-2.0-flash"
)

@dataclass
class EvaluationFeedback():
    feedback : str
    score : Literal["pass","needs_improvement","fail"]

evaluator_agent = Agent(
    name = "Evaluator Agent",
    instructions = "You are an evaluator agent. You will be given a story outline and you will evaluate it and gives feedback whether it needs to be improved or not and also gives score one out of these 'pass','needs_improvement, 'fails' ",
    model = "gemini-2.0-flash",
    output_type = EvaluationFeedback,
)

async def main():
    msg = input("Enter a story idea: ")
    input_items : list[TResponseInputItem] = [{"role":"user","content":msg}]
    outline_generated = None
    while True:
        outline_result = await Runner.run(
            story_outline_generator,
            input_items,
        )
        input_items = outline_result.to_input_list()
        outline_generated = ItemHelpers.text_message_outputs(outline_result.new_items)
        print(f"Outline generated: {outline_generated}")

        evaluation_result = await Runner.run(
            evaluator_agent,
            input_items
        )
        evaluator_output : EvaluationFeedback = evaluation_result.final_output
        print(f"Evaluation feedback: {evaluator_output.feedback}")
        print(f"Score : {evaluator_output.score}")

        if evaluator_output.score == "pass":
            print("Outline is good to go!")
            break

        print("Outline needs improvement. Generating a new outline...")
        input_items.append({"role":"assistant","content":f"Feedback: {evaluator_output.feedback} Score:{evaluator_output.score}"})
    print("Final outline:", outline_generated)

asyncio.run(main())
# This code is a simple example of how to use the agents to generate a story outline and evaluate it.

