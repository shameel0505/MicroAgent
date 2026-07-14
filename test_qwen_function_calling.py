import json
import requests

def test_function_calling():
    print("Testing function calling with qwen2.5-coder:7b in Ollama...")
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": "qwen2.5-coder:7b",
        "messages": [
            {"role": "user", "content": "What is the weather like in Paris? Please use celsius."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print("\nResponse from Ollama:")
        print(json.dumps(result, indent=2))
        
        message = result.get("message", {})
        if "tool_calls" in message:
            print("\n✅ SUCCESS: The model correctly decided to use a tool!")
            for tool_call in message["tool_calls"]:
                print(f"Tool Name: {tool_call['function']['name']}")
                print(f"Arguments: {tool_call['function']['arguments']}")
        else:
            print("\n❌ FAILURE: The model did not use any tools.")
            
    except Exception as e:
        print(f"Error making request to Ollama: {e}")

if __name__ == "__main__":
    test_function_calling()
