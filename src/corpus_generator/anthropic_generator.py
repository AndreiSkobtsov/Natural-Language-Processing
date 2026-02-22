from .base_generator import TextGenerator
class AnthropicGenerator(TextGenerator):
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        super().__init__(model_name, temperature)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text