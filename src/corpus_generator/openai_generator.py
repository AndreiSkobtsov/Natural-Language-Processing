from openai import OpenAI

class OpenAIGenerator:
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return ""