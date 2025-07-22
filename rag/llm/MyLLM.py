# MyLLM.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class MyLLM:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "deepseek/deepseek-chat-v3-0324:free"
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def ask(self, prompt: str, specialization: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": specialization},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
        }

        response = requests.post(self.url, headers=self.headers, json=body)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"].strip()