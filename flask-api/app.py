import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/api/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy", "message": "Chatbot API is running"})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7101))
    app.run(host='0.0.0.0', port=port, debug=True) 