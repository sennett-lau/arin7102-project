import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import openai
import json
import ast
import pandas as pd
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS to explicitly allow all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure OpenAI
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
client = OpenAI(api_key=api_key,base_url=base_url)

# 导入医疗推荐工具
try:
    from mcp_tools.medicine_recommender import recommend_medicine, get_all_conditions, get_all_age_groups
    # 加载医疗数据
    medicine_df = pd.read_csv("../src/data/webmd.csv", encoding="utf-8")
except ImportError:
    print("警告: 医疗推荐工具未找到，相关功能可能不可用")
    medicine_df = None

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
    if medicine_df is not None:
        df = medicine_df
        medicine_recommendation = recommend_medicine(df, age, condition, sex)
        #print("medicine_recommendation==========>", medicine_recommendation)
        return json.dumps(medicine_recommendation)
    else:
        return json.dumps({"error": "医疗推荐功能不可用"})

# 定义工具
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
    }
]

# 如果医疗数据可用，添加医疗推荐工具
if medicine_df is not None:
    tools.append({
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
    })

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy", "message": "Chatbot API is running"})

@app.route('/api/chat', methods=['POST'])
def chat():
    settings = {
        "model": "deepseek-chat",
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
    }
    try:
        data = request.get_json()
        #print("data==========>",data)
        if not data or 'message' not in data:
            return jsonify({"error": "Missing message parameter"}), 400
        
        user_message = data['message']
        conversation_history = data.get('history', [])
        
        # Prepare messages for OpenAI API
        messages = []
        
        # Add conversation history if provided
        for entry in conversation_history:
            messages.append({
                "role": entry["role"],
                "content": entry["content"]
            })
        
        # Add the current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Call OpenAI API with tools
        response = client.chat.completions.create(
            messages=messages,
            **settings,
        )
        
        assistant_response = response.choices[0].message
        
        # 检查是否有工具调用
        if hasattr(assistant_response, 'tool_calls') and assistant_response.tool_calls:
            tool_call = assistant_response.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # 执行相应的工具函数
            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit")
                )
            elif function_name == "medicine_recommendation" and medicine_df is not None:
                function_response = medicine_recommendation(
                    function_args.get("age"), 
                    function_args.get("condition"), 
                    function_args.get("sex")
                )
            else:
                function_response = json.dumps({"error": "未知或不可用的工具函数"})
            
            # 将工具响应添加到消息历史
            messages.append(assistant_response.model_dump())
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": function_response,
                "tool_call_id": tool_call.id
            })
            
            # 再次调用API以获取最终响应
            second_response = client.chat.completions.create(
                messages=messages,
                **settings,
            )
            #print("second_response==========>",second_response)
            assistant_message = second_response.choices[0].message.content
        else:
            assistant_message = assistant_response.content
        
        return jsonify({
            "response": assistant_message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tools', methods=['GET'])
def get_available_tools():
    """返回可用工具列表的端点"""
    return jsonify({
        "tools": tools
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7101))
    app.run(host='0.0.0.0', port=port, debug=True) 