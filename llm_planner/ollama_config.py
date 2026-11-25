import json
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

URL_OLLAMA = 'URL_OLLAMA'
USER_ID_OLLAMA = 'WRITE USER ID HERE'