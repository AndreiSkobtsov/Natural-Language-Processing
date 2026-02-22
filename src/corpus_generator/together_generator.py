import requests
import json

class TogetherGenerator:
    """Uses Together AI API for LLaMA and other open models."""
    
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  Together API error: {e}")
            return ""