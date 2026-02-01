import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Importing the required libraries and modules
from agents import Agent , Runner , AsyncOpenAI , OpenAIChatCompletionsModel, InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
from agents.run import RunConfig
from pydantic import BaseModel , Field

external_client = AsyncOpenAI(
    api_key = os.getenv("GOOGLE_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client,
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True,
)

# Starting To Make An agent

# First We Impose A Guardrail Checkpoint On The Agent To Check If The Given Text Is A Homework Or Not
# This is a very important step as we need to check if the given text is a homework or not
# If it is a homework then we need to handoff the task to the respective agent

class HomeworkOutput(BaseModel):
    is_homework : bool = Field(description="Is the given text a homework?")
    reasoning : str = Field(description="Reasoning Behind your answer")

# Now Making The Guardrail Checkpoint Agent
guardrail_agent = Agent(
    name = "Guardrail Agent",
    instructions = "You are a Guardrail agent. You are very good at checking if the given text is a homework or not. If it is a homework then you need to handoff the task to the respective agent.",
    model = model,
    output_type = HomeworkOutput,
)

# Now Our Tutor Agents
physics_tutor_agent = Agent(
    name = "Physics Tutor",
    handoff_description = "You are a very adept physics tutor. You are very good at answering questions about physics. You are also very good at explaining the concepts behind the answers. You are also very good at giving hints to help students find the answers themselves.",
    model = model,
)

math_tutor_agent = Agent(
    name = "Math Tutor",
    handoff_description = "You are a very adept math tutor. You are very good at answering questions about math. You are also very good at explaining the concepts behind the answers. You are also very good at giving hints to help students find the answers themselves.",
    model = model,
)

biology_tutor_agent = Agent(
    name = "Biology Tutor",
    handoff_description = "You are a very adept biology tutor. You are very good at answering questions about biology. You are also very good at explaining the concepts behind the answers. You are also very good at giving hints to help students find the answers themselves.",
    model = model,
)

english_tutor_agent = Agent(
    name = "English Tutor",
    handoff_description = "You are a very adept English tutor. You are very good at answering questions about English. You are also very good at explaining the concepts behind the answers.",
    model = model,
)

islamiat_tutor_agent = Agent(
    name = "Islamiat Tutor",
    handoff_description = "You are a very adept Islamiat tutor. You are very good at answering questions about Islamiat which means the study and history of Islam, Muslims and Quran.",
    model = model,
)

pak_studies_tutor_agent = Agent(
    name = "Pak Studies Tutor",
    handoff_description = "You are a very adept Pak Studies tutor. You are very good at answering questions about Pakistan Studies which means the study of Pakistan, its history, geography and culture.",
    model = model,
)

computer_science_tutor_agent = Agent(
    name = "Computer Science Tutor",
    handoff_description = "You are a very adept Computer Science tutor. You are very good at answering questions about Computer Science. You are also very good at explaining the concepts behind the answers.",
    model = model,
)

chemistry_tutor_agent = Agent(
    name = "Chemistry Tutor",
    handoff_description = "You are a very adept Chemistry tutor. You are very good at answering questions about Chemistry. You are also very good at explaining the concepts behind the answers.",
    model = model,
)

history_tutor_agent = Agent(
    name = "History Tutor",
    handoff_description = "You are a very adept History tutor. You are very good at answering questions about History. You are also very good at explaining the concepts behind the answers.",
    model = model,
)

# Now we Make Our Guardrail Function To Pass It into our Decision Maker Agent Below To Check whether the User Query is about Homework Or Not based on the result to decide whether to pass the query to respective tutor or not
async def homework_Guardrail(ctx,agent,input_data):
    result = await Runner.run(
        guardrail_agent,
        input_data,
        context = ctx.context,
        run_config = config,
    )
    final_output = result.final_output_as(HomeworkOutput)
    print(f"Is the given text a homework? {final_output.is_homework}")
    print(f"Reasoning Behind your answer: {final_output.reasoning}")
    # Now we need to check if the given text is a homework or not
    return GuardrailFunctionOutput(
        output_info = final_output,
        tripwire_triggered = not final_output.is_homework,
    )

# Now Making Our Decision Maker Agent Or Can Also Be Called As Handoff Agent
decision_maker_agent = Agent(
    name = "Decision Maker Agent",
    instructions = "You are a decision maker agent. You are very good at deciding which agent to handoff the task to based on the user HomeWork Question.",
    model = model,
    handoffs = [physics_tutor_agent , math_tutor_agent , biology_tutor_agent , english_tutor_agent , islamiat_tutor_agent , pak_studies_tutor_agent , computer_science_tutor_agent , chemistry_tutor_agent, history_tutor_agent],
    input_guardrails = [
        InputGuardrail(
            guardrail_function = homework_Guardrail,
        )
    ]
)

async def main():
    try:
        # Now we need to call the decision maker agent with the user query
        result_1 = await Runner.run(
            decision_maker_agent,
            "Recommend me a movie?",
            run_config = config,
        )
        print("Agent is called for result_1")
        print(result_1.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("The Following Query Does Not Represent A Homework , please try again")
    try:
        result_2 = await Runner.run(
            decision_maker_agent,
            "What is the speed of light and why it is of that value?",
            run_config = config,
        )
        print("Agent is called for result_2")
        print(result_2.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("The Following Query Does Not Represent A Homework , please try again")
    
    try:
        result_3 = await Runner.run(
            decision_maker_agent,
            "What is the formula of water?",
            run_config = config,
        )
        print("Agent is called for result_3")
        print(result_3.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("The Following Query Does Not Represent A Homework , please try again")
# Now Running the whole code
asyncio.run(main())