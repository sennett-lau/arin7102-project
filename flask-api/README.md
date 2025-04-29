# Chatbot Flask Backend

A simple Flask backend for a chatbot application.

## Setup

### Option 1: Standard Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the application:
```
python app.py
```

### Option 2: Docker Setup

1. Build the Docker image:
```
docker build -t arin7102-project-flask-api .
```

2. Run the Docker container:
```
docker run -p 7101:7101 arin7102-project-flask-api
```

The server will start on port 7101 (or the port specified in the .env file).

## API Endpoints

- `GET /api/healthcheck`: Check if the API is running
- `POST /api/chat`: Send a message to the chatbot

### Chat API Usage

Request body:
```json
{
  "message": "Your message here",
  "history": [
    {"role": "user", "content": "Previous user message"},
    {"role": "assistant", "content": "Previous assistant response"}
  ]
}
```

Response:
```json
{
  "response": "Assistant's response"
}
```

## Configuration

Set your OpenAI API key in the `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
``` 