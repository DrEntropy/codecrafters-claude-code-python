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
    elif name == "Write":
        file_path = arguments.get("file_path")
        content = arguments.get("content")
        if not file_path or content is None:
            return "Error: 'file_path' and 'content' arguments are required"
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return "Write successful"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    else:
        return f"Unknown tool: {name}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


    read_tool = {
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
    
    write_tool = {
                "type": "function",
                "function": {
                    "name": "Write",
                    "description": "Write content to a file",
                    "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to"
                        },
                        "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                        }
                    }
                    }
                }
                }
    
    finished = False
    messages = [{"role": "user", "content": args.p}]

   # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)
    
    # This loop will continue until the model stops making tool calls
    loop_max = 5 # just in case, to prevent infinite loops
    while not finished:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=[read_tool, write_tool]
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")



        #print(chat.choices[0].message.content)
        message = chat.choices[0].message
        messages.append(message)

        if message.content:
            # uncomment to see all responses.
            #print(message.content)
            pass 
    
        if message.tool_calls is not None and len(message.tool_calls) > 0:
            tool_call = chat.choices[0].message.tool_calls[0]
            if tool_call.type == "function":
                function = tool_call.function
                id = tool_call.id
                name = function.name
                arguments = json.loads(function.arguments)
                print(f"Tool call: {name} with arguments {arguments}", file=sys.stderr)
                result = dispatch_tool_call(name, arguments)
            else:
                print(f"Unknown tool call type: {tool_call.type}", file=sys.stderr)
                result = f"Error: Unknown tool call type {tool_call.type}"
            tool_call_result_message = {
                "role": "tool",
                "tool_call_id" : id,
                "content": result,
            }
            messages.append(tool_call_result_message)
        else:
            # no tool calls, we're done
            finished = True
    
        loop_max -= 1
        if loop_max <= 0:
            print("Reached maximum loop count, stopping to prevent infinite loop.", file=sys.stderr)
            break

    # return only the final content message
    if message.content:     
        print(message.content)


if __name__ == "__main__":
    main()
