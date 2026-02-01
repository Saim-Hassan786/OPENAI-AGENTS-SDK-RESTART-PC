import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    ItemHelpers,
    trace,
    set_trace_processors,
    set_tracing_disabled
)
import asyncio
from openai import AsyncOpenAI
from agents.tracing.processor_interface import TracingProcessor
from pprint import pprint

external_client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
set_default_openai_client(client=external_client)
set_default_openai_api("chat_completions")

# The Code Below Is For Tracing Use This Tracing Processor If You Want To Enable Tracing
# Uncomment the code below to enable local tracing

# class LocalTracingProcessor(TracingProcessor):
#     def __init__(self):
#         self.traces = []
#         self.spans = []
    
#     def on_trace_start(self, trace):
#         self.traces.append(trace)
#         print(f"Trace started: {trace.trace_id}")
    
#     def on_span_start(self, span):
#         self.spans.append(span)
#         print(f"Span started: {span.span_id}")
#         pprint(span.export())

#     def on_span_end(self, span):
#         print(f"Span ended: {span.span_id}")
#         pprint(span.export())

#     def on_trace_end(self, trace):
#         print(f"Trace ended: {trace.trace_id}")
#         pprint(trace.export())
        
#     def force_flush(self):
#         print("Flushing traces and spans")
    
#     def shutdown(self):
#         print("======Shutting down tracing processor======")
#         print("------Collected Traces------:")
#         for trace in self.traces:
#             pprint(trace.export())
#         print("------Collected Spans------:")
#         for span in self.spans:
#             pprint(span.export())

# local_tracing = LocalTracingProcessor()
# set_trace_processors([local_tracing])

# The Code Below Is For Parallelization
spanish_translator_agent = Agent(
    name="Spanish Translator",
    instructions="Translate the given text to Spanish",
    model = "gemini-2.0-flash",
)

best_translation_picker_agent = Agent(
    name="Best Translation Picker",
    instructions="Given a list of translations, pick the best one based on accuracy and fluency.",
    model = "gemini-2.0-flash",
)

set_tracing_disabled(True)  # Disable tracing for parallel execution

# The Code Below Is For Running the Agents in Parallel
async def main():
    user_input = input("Enter the text to translate: ")
    # with trace("Translation Tracing"):
    result_1,result_2,result_3 = await asyncio.gather(
            Runner.run(spanish_translator_agent,user_input),
            Runner.run(spanish_translator_agent,user_input),
            Runner.run(spanish_translator_agent,user_input)
        )
    print(f"\n\nTranslation Results:\n\n Runner1\n\n{result_1.final_output}\n\nRunner2\n\n{result_2.final_output}\n\nRunner3\n\n{result_3.final_output}")
    results = [
            ItemHelpers.text_message_outputs(result_1.new_items),
            ItemHelpers.text_message_outputs(result_2.new_items),
            ItemHelpers.text_message_outputs(result_3.new_items)
        ]
    translations = "\n\n".join(results)
    print(f"\n\nCompiled Translation Results:\n\n{translations}")

    best_selected_translation = await Runner.run(
            best_translation_picker_agent,
            input=f"Input: {user_input}\n\nTranslations:\n{translations}"
            )
    print(f"\n\nBest Selected Translation:\n\n{best_selected_translation.final_output}")

asyncio.run(main())