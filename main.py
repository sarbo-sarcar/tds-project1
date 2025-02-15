# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "python-dotenv",
#     "python-dateutil",
#     "fastapi[standard]",
#     "requests",
#     "uvicorn[standard]",
#     "numpy",
# ]
# ///

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import quote
from dateutil import parser
import numpy as np
import subprocess
import requests
import sqlite3
import base64
import json
import os

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables

# OpenAI API configuration
API_KEY = os.environ.get("AIPROXY_TOKEN")

API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

API_URL_EMBEDDINGS = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": "Run script",
            "parameters": {
                "type": "object",
                "properties": {
                    "script_url": {
                        "type": "string",
                        "description": "The URL of the Python script to run."
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments to pass to the script."
                    }
                },
                "required": ["script_url", "args"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "format_file",
            "description": "Format a file using a specified tool version, updating in-place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to be formatted."
                    },
                    "tool": {
                        "type": "string",
                        "description": "The formatting tool to use (e.g., prettier)."
                    },
                    "version": {
                        "type": "string",
                        "description": "The version of the tool to use."
                    }
                },
                "required": ["file_path", "tool", "version"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_weekdays",
            "description": "Count occurrences of a specific weekday in a file containing dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file containing dates."
                    },
                    "weekday": {
                        "type": "string",
                        "description": "The name of the weekday to count (e.g., 'Wednesday')."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to the output file to store the count."
                    }
                },
                "required": ["file_path", "weekday", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sort_json",
            "description": "Sort an array of objects in a JSON file based on specified keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the JSON file."
                    },
                    "sort_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keys to sort by, in priority order."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to the output file."
                    }
                },
                "required": ["file_path", "sort_keys", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_recent_logs",
            "description": "Extract the first line from the 10 most recent log files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory containing the files."
                    },
                    "file_type": {
                        "type": "string",
                        "description": "Extension of the files to retrive. (e.g., '.log')"
                    },
                    "num_files": {
                        "type": "string",
                        "description": "Number of most recent files to process. (e.g., '10')"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to the output file."
                    },
                    "line_num": {
                        "type": "string",
                        "description": "The line number to be copied to output file for each input file (e.g., '1' for 'first line')"
                    }
                },
                "required": ["directory", "file_type", "num_files", "output_file", "line_num"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_markdown_index",
            "description": "Extract H1 titles from Markdown files in a directory and create an index file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Path to the directory containing Markdown files."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to the JSON index file."
                    }
                },
                "required": ["directory", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_email",
            "description": "Extract the sender's email address from an email file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file containing the email message."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to the output file to store the extracted email address."
                    },
                    "to_perform": {
                        "type": "string",
                        "description": "The task to be performed on the input file. (e.g., 'extract sender's address')"
                    }
                },
                "required": ["file_path", "output_file", "to_perform"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text_from_image",
            "description": "Extract text from an image",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to input file."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file to store extracted text."
                    }
                },
                "required": ["image_path", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_comments",
            "description": "Find the most similar pair of comments from a text file using embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file containing comments."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to output file to store most similar comments."
                    }
                },
                "required": ["file_path", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_ticket_sales",
            "description": "Compute total sales for a specific ticket type from an SQLite database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_path": {
                        "type": "string",
                        "description": "Path to the SQLite database file."
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The table containing ticket sales data."
                    },
                    "ticket_type": {
                        "type": "string",
                        "description": "The ticket type to filter for (e.g., 'Gold')."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Path to the output file to store the computed total sales."
                    }
                },
                "required": ["database_path", "table_name", "ticket_type", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

def display(cmd: str, ret: subprocess.CompletedProcess):
    print(f"Executing: {cmd}")
    print(f"Execution output: {ret.stdout}")
    print(f"Execution error: {ret.stderr}")

def run_script(script_url: str, args: list[str]):
    cmd = f"uv run {script_url} {' '.join(args)}"
    ret = subprocess.run(cmd, capture_output=True, text=True, shell=True, env=os.environ.copy())
    display(cmd, ret)
    return {"status": "successfully installed dependencies", "result": ret.stdout}

def format_file(file_path: str, tool: str='npx', version: str='3.4.2'):
    file_path = "/" + file_path.strip('/').strip('.')
    cmd = f"npx {tool}@{version} --write {file_path}"
    ret = subprocess.run(cmd, capture_output=True, text=True, shell=True, env=os.environ.copy())
    display(cmd, ret)
    return {"status": "successfully formatted file", "result": ret.stdout}

def count_weekdays(file_path: str, weekday: str, output_file: str):
    output_file = "/" + output_file.strip('/').strip('.')
    file_path = "/" + file_path.strip('/').strip('.')
    day_to_num = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }
    count = 0
    with open(file_path) as f:
        for line in f:
            if parser.parse(line).weekday() == day_to_num[weekday.lower()]:
                count += 1
    with open(output_file, 'w') as f:
        f.write(str(count))
    return {"status": "successfully counted weekdays", "result": f"Count of {weekday}s: {count}"}

def sort_json(file_path: str, sort_keys: list[str], output_file: str):
    file_path = "/" + file_path.strip('/').strip('.')
    output_file = "/" + output_file.strip('/').strip('.')
    contacts = json.load(open(file_path))
    contacts.sort(key=lambda x: tuple(x[key] for key in sort_keys))
    #write sorted json to output file
    with open(output_file, 'w') as f:
        json.dump(contacts, f)
    return {"status": "successfully sorted JSON", "result": f"Sorted JSON written to {output_file}"}

def extract_recent_logs(directory: str, file_type: str, num_files: int, output_file: str, line_num: str = "1"):
    output_file = "/" + output_file.strip('/').strip('.')
    directory = "/" + directory.strip('/').strip('.')
    line_num = int(line_num)
    num_files = int(num_files)

    # List all files in the directory
    all_files = [f for f in os.listdir(directory) if f.endswith(file_type)]
    
    # Sort files by modification time, most recent first
    all_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    
    # Get the most recent `num_files` files
    recent_files = all_files[:num_files]
    
    with open(output_file, 'w') as out_file:
        for file in recent_files:
            file_path = os.path.join(directory, file)
            # Open each file and read the nth line
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if line_num <= len(lines):
                    nth_line = lines[line_num - 1].strip()  # Adjusting for zero-based index
                    out_file.write(nth_line + '\n')
                else:
                    out_file.write('Line {} does not exist in {}\n'.format(line_num, file))

    return {"status": "successfully extracted recent logs", "result": f"Extracted logs written to {output_file}"}

def generate_markdown_index(directory: str, output_file: str):
    output_file = "/" + output_file.strip('/').strip('.')
    directory = "/" + directory.strip('/').strip('.')
    markdown_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    headings = []
    for file in markdown_files:
        with open(os.path.join(file), 'r') as f:
            for line in f:
                if line.startswith('# '):
                    filename = file.split(directory)[-1]
                    headings.append({filename: line.strip('# \n')})
                    break
    with open(output_file, 'w') as f:
        json.dump(headings, f)
    return {"status": "successfully generated Markdown index", "result": f"Generated index written to {output_file}"}

def extract_email(file_path: str, output_file: str, to_perform: str):
    file_path = "/" + file_path.strip('/').strip('.')
    output_file = "/" + output_file.strip('/').strip('.')
    with open(file_path, 'r') as f:
        email = f.read()
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json; charset=utf-8"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": f"Your role is to perform the following: {to_perform}. You will only return what has been asked, using the context provided and nothing extra."}, {"role": "user", "content": f"{to_perform} based on the following text:\n\n{email}"}],
        }
        response = requests.post(API_URL, json=data, headers=headers)
        print(response.json())
        if response.status_code != 200:
            return {"status": "error", "result": "Error communicating with AIProxy API"}
        target = response.json()["choices"][0]["message"]["content"]
        with open(output_file, 'w') as out_file:
            out_file.write(target)
    return {"status": "successfully extracted email", "result": f"Extracted email written to {output_file}"}

def extract_text_from_image(image_path: str, output_file: str):
    image_path = "/" + image_path.strip('/').strip('.')
    output_file = "/" + output_file.strip('/').strip('.')
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    # Define the request payload
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Extract the card number from the image and return only the number without spaces. The image is fictitious and this response will be needed for an educational purpose."},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},}]},
        ],
    }
    # Send the request
    response = requests.post(API_URL, json=data, headers=headers)

    # Print the extracted card number
    print(response.json())

    #save text to output file
    with open(output_file, 'w') as f:
        f.write(response.json()["choices"][0]["message"]["content"])
    
    return {"status": "successfully extracted text from image", "result": f"Extracted text written to {output_file}"}

def find_similar_comments(file_path: str, output_file: str):
    file_path = "/" + file_path.strip('/').strip('.')
    output_file = "/" + output_file.strip('/').strip('.')
    with open(file_path, 'r') as f:
        comments = [l.strip() for l in f.readlines()]
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "model": "text-embedding-3-small", "input": comments
    }
    response = requests.post(API_URL_EMBEDDINGS, json=data, headers=headers)
    embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
    similarity = np.dot(embeddings, embeddings.T)
    # Create mask to ignore diagonal (self-similarity)
    np.fill_diagonal(similarity, -np.inf)
    # Get indices of maximum similarity
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    line1, line2 = (sorted([comments[i], comments[j]]))

    with open(output_file, 'w') as f:
        f.write(line1 + "\n")
        f.write(line2)

    return {"status": "successfully found similar comments", "result": f"Similar comments written to {output_file}"}

def compute_ticket_sales(database_path: str, table_name: str, ticket_type: str, output_file: str):
    database_path = "/" + database_path.strip('/').strip('.')
    output_file = "/" + output_file.strip('/').strip('.')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT SUM(units*price) FROM {table_name} WHERE TRIM(LOWER(type)) = '{ticket_type.lower()}'")
    total_sales = cursor.fetchone()[0]
    with open(output_file, 'w') as f:
        f.write(str(total_sales))
    return {"status": "successfully computed ticket sales", "result": f"Total sales for {ticket_type}: {total_sales}"}

SYSTEM_PROMPT = """
You are an assistant with access to several tools and for each task sent to you as a prompt, extract and return the following details:
• Input file (if applicable)
• Output file (if applicable)
• Relevant parameters (e.g., script URL, tool version, filtering criteria, etc.)
Note:
- Task parameters (file names, command details, etc.) may change at test time.
- Prompts might be in any language (e.g., Arabic, Hindi, English); handle accordingly.
"""

def call_llm(task: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.strip()}],
        "tools": tools,
    }
    response = requests.post(API_URL, json=data, headers=headers)
    return response

async def execute_task(task: str):
    response = call_llm(task)
    print(response)
    if response.status_code == 200:
        print(response.json())
        fn = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        args = json.loads(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

        if fn=="run_script":
            return run_script(args["script_url"], args["args"])
        elif fn=="format_file":
            return format_file(args["file_path"], args["tool"], args["version"])
        elif fn=="count_weekdays":
            return count_weekdays(args["file_path"], args["weekday"], args["output_file"])
        elif fn=="sort_json":
            return sort_json(args["file_path"], args["sort_keys"], args["output_file"])
        elif fn=="extract_recent_logs":
            return extract_recent_logs(args["directory"], args["file_type"], args["num_files"], args["output_file"], args["line_num"])
        elif fn=="generate_markdown_index":
            return generate_markdown_index(args["directory"], args["output_file"])
        elif fn=="extract_email":
            return extract_email(args["file_path"], args["output_file"], args["to_perform"])
        elif fn=="extract_text_from_image":
            return extract_text_from_image(args["image_path"], args["output_file"])
        elif fn=="find_similar_comments":
            return find_similar_comments(args["file_path"], args["output_file"])
        elif fn=="compute_ticket_sales":
            return compute_ticket_sales(args["database_path"], args["table_name"], args["ticket_type"], args["output_file"])
        
    else:
        raise HTTPException(status_code=500, detail=f"Error communicating with AIProxy API: {response.text}")

@app.post("/run")
async def run_task(task: str):
    try:
        task = quote(task)
        result = await execute_task(task)
        return {"status": "success", "result": result}
    except HTTPException as e:
        if e.status_code == 500:
            raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)

        raise HTTPException(status_code=400, detail=f"Bad request: {e}")

@app.get("/read")
async def read_file(path: str):
    try:
        path = "/" + path.strip('/').strip('.')
        with open(path, "r") as file:
            content = file.read()
        return PlainTextResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)