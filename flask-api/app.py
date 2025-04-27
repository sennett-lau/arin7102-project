import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Configure CORS to explicitly allow all origins
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy", "message": "Chatbot API is running"})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
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
        
        # Call OpenAI API
        response = openai.chat.completions.create(
            model="o4-mini",
            messages=messages
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({
            "response": assistant_message
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7101))
    app.run(host='0.0.0.0', port=port, debug=True) 