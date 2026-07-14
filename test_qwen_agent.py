import os
import sys
import contextlib
import io
import json
import re
import requests

def execute_python_code(code: str) -> str:
    """Executes Python code locally and captures the output."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            exec(code, globals())
        except Exception as e:
            return f"Error executing code: {e}"
    
    result = output.getvalue()
    return result if result else "Code executed successfully but printed nothing."

def chat_with_ollama(messages, model_name):
    """Sends the message history to Ollama and yields the streamed text."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        full_text = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                full_text += chunk
                yield chunk
    except Exception as e:
        yield f"\n[Error communicating with Ollama: {e}]"

def main():
    print("=====================================================")
    print("🤖 Agent LLM Arena (100% Bulletproof)")
    print("=====================================================\n")
    
    print("Which model would you like to use?")
    print("1. qwen2.5-coder:7b")
    print("2. gemma4:12b")
    choice = input("Enter 1 or 2: ")
    
    model_name = "qwen2.5-coder:7b" if choice.strip() == "1" else "gemma4:12b"
    print(f"\n[Booting up {model_name}...]\n")
    
    system_instruction = '''You are a powerful local AI assistant on a Mac. 
You have access to a tool called `local_python_executor`. 
To execute code on the user's machine, you MUST output exactly this special tag format and NOTHING ELSE:

<{"name": "local_python_executor", "arguments": {"code": "import os\\nprint(os.listdir('/Users/shameel/Desktop'))"}}>

Do not write markdown ```json blocks. Just output the tag.
'''

    messages = [
        {"role": "system", "content": system_instruction}
    ]
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not user_input.strip():
                continue
                
            messages.append({'role': 'user', 'content': user_input})
            print(f"\n{model_name}: ", end="", flush=True)
            
            while True:
                # Stream the response from Ollama
                ai_reply = ""
                for chunk in chat_with_ollama(messages, model_name):
                    print(chunk, end="", flush=True)
                    ai_reply += chunk
                
                messages.append({"role": "assistant", "content": ai_reply})
                
                # Check if AI wants to execute code
                if '<{"name":' in ai_reply:
                    match = re.search(r'<({.*?})>', ai_reply, re.DOTALL)
                    if match:
                        try:
                            tool_json = json.loads(match.group(1))
                            if tool_json.get('name') == 'local_python_executor':
                                print(f"\n\n[🛠️  Executing Python Code locally...]")
                                code = tool_json.get('arguments', {}).get('code', '')
                                result = execute_python_code(code)
                                print(f"[Result]:\n{result}")
                                
                                # Feed the result back to the model so it can summarize
                                messages.append({
                                    'role': 'user', 
                                    'content': f"Observation from local_python_executor:\n{result}\nNow summarize this clearly to the user."
                                })
                                print(f"\n{model_name} (Summarizing): ", end="", flush=True)
                                continue # Run the LLM again with the new observation
                        except Exception as e:
                            print(f"\n[❌ Failed to parse or run intercepted tool: {e}]")
                
                # If no tool was called, we're done with this turn
                print()
                break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break

if __name__ == '__main__':
    main()
