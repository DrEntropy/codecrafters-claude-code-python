import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

def dispatch_tool_call(name, arguments):
    if name == "Read":
        file_path = arguments.get("file_path")
        if not file_path:
            return "Error: 'file_path' argument is required"
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    else:
        return f"Unknown tool: {name}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


    tools = {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
                    }
                }
                }

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[tools]
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the following line to pass the first stage
    print(chat.choices[0].message.content)
 
    if len(chat.choices[0].message.tool_calls) > 0:
        tool_call = chat.choices[0].message.tool_calls[0]
        if tool_call.type == "function":
            function = tool_call.function
            name = function.name
            arguments = json.loads(function.arguments)
            print(f"Tool call: {name} with arguments {arguments}", file=sys.stderr)
            result = dispatch_tool_call(name, arguments)
        else:
            print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
            result = f"Error: Unknown tool call type {tool_call.type}"
        print(f"{result}")
    
    


if __name__ == "__main__":
    main()
