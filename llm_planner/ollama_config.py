import json
import os
import re
import requests

def get_response(input_string):
    output_string = re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    return output_string

DEFAULT_MODEL = "deepseek-r1:14b"

def chat_with_model(query,url,user_id,model=DEFAULT_MODEL,):

    headers = {
        'Authorization': user_id,
        'Content-Type': 'application/json'
    }
    data = {
      "model": model,
      "messages": [
        {
          "role": "user",
          "content": query
        }
      ]
    }
    json_data = json.dumps(data)
    response = requests.post(url, headers=headers, data=json_data)
    result = response.json()
    return result

# The endpoint and its token are read from the environment, so no credential lives in
# the repository. Set OLLAMA_URL and OLLAMA_TOKEN before running a demo.
URL_OLLAMA = os.environ.get("OLLAMA_URL", "")
USER_ID_OLLAMA = os.environ.get("OLLAMA_TOKEN", "")
