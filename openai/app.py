import json
import ast
import os
from openai import AsyncOpenAI
import pandas as pd
import chainlit as cl

from mcp_tools.medicine_recommender import recommend_medicine,get_all_conditions,get_all_age_groups

cl.instrument_openai()

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")

client = AsyncOpenAI(api_key=api_key,base_url=base_url)

MAX_ITER = 5

medicine_df = pd.read_csv("D:/Work/arin7102-project/webmd.csv",encoding="utf-8")

def get_current_weather(location, unit):
    unit = unit or "Fahrenheit"
    weather_info = {
        "location": location,
        "temperature": "60",
        "unit": unit,
        "forecast": ["windy"],
    }
    return json.dumps(weather_info)

def medicine_recommendation(age, condition, sex):
    df = medicine_df
    medicine_recommendation = recommend_medicine(df, age, condition, sex)
    print("medicine_recommendation==========",medicine_recommendation)
    return json.dumps(medicine_recommendation)


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },  
    },
    {
        "type": "function",
        "function": {
            "name": "medicine_recommendation",
            "description": "Recommend medicine based on age, condition, and sex",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "string", "description": "The age of the patient", "enum": get_all_age_groups(medicine_df)},
                    "condition": {"type": "string", "description": "The condition of the patient, Must be English."},
                    "sex": {"type": "string", "description": "The sex of the patient", "enum": ["Male", "Female"]},
                },
                "required": ["age", "condition", "sex"],
            },
        },
    }   
]


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.step(type="tool")
async def call_tool(tool_call_id, name, arguments, message_history):
    arguments = ast.literal_eval(arguments)
    print("message_history==========",message_history)
    current_step = cl.context.current_step
    current_step.name = name
    current_step.input = arguments

    if name == "get_current_weather":
        function_response = get_current_weather(
            location=arguments.get("location"),
            unit=arguments.get("unit"),
        )
    elif name == "medicine_recommendation":
        function_response = medicine_recommendation(arguments.get("age"), arguments.get("condition"), arguments.get("sex"))

    current_step.output = function_response
    current_step.language = "json"

    message_history.append(
        {
            "role": "tool",
            "name": name,
            "content": function_response,
            "tool_call_id": tool_call_id,
        }
    )


async def call_gpt4(message_history):
    settings = {
        "model": "deepseek-chat",
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }
    #print("message_history==========",message_history)
    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    tool_call_id = None
    function_output = {"name": "", "arguments": ""}

    final_answer = cl.Message(content="", author="Answer")

    async for part in stream:
        new_delta = part.choices[0].delta
        tool_call = new_delta.tool_calls and new_delta.tool_calls[0]
       # print("new_delta==========",new_delta)
        #print("tool_call==========",tool_call)
        function = tool_call and tool_call.function
        if tool_call and tool_call.id:
            tool_call_id = tool_call.id

        if function:
            if function.name:
                function_output["name"] = function.name
            else:
                function_output["arguments"] += function.arguments

        if new_delta.content:
            if not final_answer.content:
                await final_answer.send()
            await final_answer.stream_token(new_delta.content)

    if tool_call_id:
        message_history.append({"role": "assistant", "content": "","tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": function_output["name"], "arguments": function_output["arguments"]}}]})
        await call_tool(
            tool_call_id,
            function_output["name"],
            function_output["arguments"],
            message_history,
        )

    if final_answer.content:
        await final_answer.update()

    return tool_call_id


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    cur_iter = 0

    while cur_iter < MAX_ITER:
        tool_call_id = await call_gpt4(message_history)
        if not tool_call_id:
            break

        cur_iter += 1