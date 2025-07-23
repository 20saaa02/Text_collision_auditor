# MyLLM.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class MyLLM:
    def __init__(self):
        self.iam_token = os.getenv("YANDEX_GPT_IAM_TOKEN")
        self.folder_id = os.getenv("YANDEX_GPT_FOLDER_ID")
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json",
        }

    def ask(self, prompt: str, specialization: str) -> str:
        body = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt",
            "completionOptions": {
                "temperature": 0.0,
                "maxTokens": 1000
            },
            "messages": [
                {"role": "system", "text": specialization},
                {"role": "user", "text": prompt}
            ],
        }

        response = requests.post(self.url, headers=self.headers, json=body)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

        return response.json()["result"]["alternatives"][0]["message"]["text"].strip()