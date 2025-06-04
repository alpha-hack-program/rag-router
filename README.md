mkdir scratch


cat >> EOF > ./scratch/model_map.json
{
  "granite-3-1-8b": {
    "url": "https://granite-3-1-8b-instruct.acme.com/v1/chat/completions",
    "model": "granite-3-1-8b-instruct",
    "api_key": "<KEY>"
  },
  "mistral-7b-instruct-v0-3": {
    "url": "https://mistral-7b-instruct-v0-3.acme.com/v1/chat/completions",
    "model": "mistral-7b-instruct",
    "api_key": "<KEY>"
  },
  "llama-3-1-8b": {
    "url": "https://llama-3-1-8b-instruct.acme.com/v1/chat/completions",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "api_key": "<KEY>"
  }
}

