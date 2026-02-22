from abc import ABC, abstractmethod
from typing import List

class TextGenerator(ABC):
    #Abstract base class for all text generators
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model_name = model_name
        self.temperature = temperature
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        #Generate a single text from a prompt
        pass
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 200) -> List[str]:
        #Generate texts for multiple prompts (default implementation loops)
        return [self.generate(p, max_tokens) for p in prompts]