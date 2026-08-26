import json
import re
import requests

def get_response(input_string):
    output_string = re.sub(r'<think>.*?</think>', '', input_string, flags=re.DOTALL)
    return output_string

def chat_with_model(query,url,user_id,model="deepseek-r1:14b",):

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

URL_OLLAMA = 'http://192.168.200.10:3000/api/chat/completions'
USER_ID_OLLAMA = 'Bearer sk-ade11fba62274285a7c52fb5862eb245'