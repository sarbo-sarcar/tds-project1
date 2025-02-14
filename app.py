# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "python-dotenv",
#     "fastapi[standard]",
#     "requests",
#     "uvicorn[standard]",
# ]
# ///

from fastapi import FastAPI, HTTPException
import os
import subprocess
from urllib.parse import quote
import requests
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


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

# """You are an agent assistance that accepts a user query in natural language and then parses it to reason and build terminal commands that achieve the objective task mentioned in the query. You will not return any chain-of-thought or explanation or reasoning explicitly. Your task is to only return a && separated list of terminal (unix) commands that help to achieve the task. You must remember that the output you produce will directly be copied to python's subprocess module's run() function. Remember to only return the set of terminal commands for achieving the natural language query without any other details. Chain multiple commands if required using &&. Your returned response must not contain any other code delimiters or backticks. Certain queries may ask for usage of a particular command which is not guaranteed to be installed (say, uv from python). In such cases you need to install that and then perform the task. For all tasks you need to assume that all read/write tasks are to be performed on files under the ./data directory where the subprocess is being run. There could be tasks that ask you to run a link <url_path>/<filename>.py with uv command for example (uv run <file>).
# You should remember that `uv` in every context refers to the `uv` package manager in python (typically used as uv run <file>).
# Moreover while using uv run on an external URL with a .py file, use an extra command line argument `--root ./data` to ensure that any file generated is kept at ./data (even if the query does not explicitly mention that).
# Always use `./` to begin any file name or path even if the query does not explicitly mention that.
# You must accordingly decide which command to use for which task. You may refer to the sample example given below for reference:
# Example:
# query: Install uv (if required) and run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py with 23f2002621@ds.study.iitm.ac.in as the only argument.
# ideal response: pip install uv && uv run https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py 23f2002621@ds.study.iitm.ac.in --root ./data"""

SYSTEM_PROMPT = """
You are an assistant that converts natural language queries into minimal executable UNIX commands. Your responses must:
Return only a &&-separated list of commands (no explanations, delimiters, or code blocks).
Use uv strictly as the Python package manager (assume it's installed).
Pip install non-default libraries.
Operate exclusively within ./data/, prefixing all file paths with ./
When running .py files from a URL via uv run, append --root ./data.
Run python from terminal with python3 -c 'commands'. (use single quotes for python code)
Handle input (always inconsistently formatted) using robust tools (e.g., dateutil.parser for dates).
Never use backslashes(\\) in any response.
Example 1:
Query: Install uv and run $url.py with $email as the only argument.
Response:
uv run $url.py $email --root ./data
Example 2:
Query: Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place
Response:
npx prettier@3.4.2 --write ./data/format.md
Example 3:
Query: The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to /data/dates-wednesdays.txt
Response:
pip install python-dateutil && python3 -c "from dateutil import parser; print(sum(1 for line in open('./data/dates.txt') if parser.parse(line).weekday() == 2))" > ./data/dates-wednesdays.txt
"""
@app.post("/test")
async def test():
    cmd = 'pip install python-dateutil && python3 -c \'from dateutil import parser; print(sum(1 for line in open("./data/dates.txt") if parser.parse(line).weekday() == 2))\' > ./data/dates-wednesdays.txt'
    ret = subprocess.run(cmd, capture_output=True, text=True, shell=True, env=os.environ.copy())
    print(f"Executing: {cmd}")
    print(f"Execution output: {ret.stdout}")
    print(f"Execution error: {ret.stderr}")
    return {"status": "success", "result": ret.stdout}

async def execute_task(task: str):
    print("Function called")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.strip()}]
    }
    response = requests.post(API_URL, json=data, headers=headers)
    print(response)
    if response.status_code == 200:
        print(response.json())
        cmd = response.json()["choices"][0]["message"]["content"]
        ret = subprocess.run(cmd, capture_output=True, text=True, shell=True, env=os.environ.copy())
        print(f"Executing: {cmd}")
        print(f"Execution output: {ret.stdout}")
        print(f"Execution error: {ret.stderr}")
        return cmd
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
        path = "./" + path.strip('/').strip('.')
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

