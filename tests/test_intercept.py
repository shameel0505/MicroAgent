import re
import json

content = """<{"name": "local_python_executor", "arguments": {"code": "import os\\nprint(os.listdir('/Users/shameel/Desktop'))"}}>"""

if '<{"name":' in content:
    print("Intercepting!")
    match = re.search(r'<({.*?})>', content, re.DOTALL)
    if match:
        print("Match found!")
        try:
            tool_json = json.loads(match.group(1))
            print("Parsed:", tool_json)
        except Exception as e:
            print("JSON parse error:", e)
    else:
        print("Regex match failed!")
else:
    print("Condition failed!")
