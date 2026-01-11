"""
vLLM interaction utilities.

Handles communication with vLLM server for generation and LoRA management.
"""

import requests
from pathlib import Path


class VLLMClient:
    """Client for interacting with vLLM server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.loaded_loras = []
    
    def check_connection(self) -> bool:
        """Check if vLLM server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            print(f"Connected to vLLM. Serving: {[m['id'] for m in data['data']]}")
            return True
        except Exception as e:
            print(f"[ERROR] Could not connect to vLLM at {self.base_url}")
            print(f"Error details: {e}")
            return False
    
    def generate(
        self,
        prompts: list[str],
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0,
        n: int = 1,
    ) -> list[str]:
        """
        Generate completions for a batch of prompts.
        
        Args:
            prompts: List of prompt strings
            model_id: Model or LoRA adapter name to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            n: Number of completions per prompt
            
        Returns:
            List of generated text strings
        """
        api_url = f"{self.base_url}/v1/completions"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "model": model_id,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        outputs = [choice["text"] for choice in result["choices"]]
        return outputs
    
    def load_lora(self, lora_name: str, lora_path: str = None) -> bool:
        """
        Load a LoRA adapter into vLLM.
        
        Args:
            lora_name: Name identifier for the LoRA
            lora_path: Absolute path to LoRA weights (defaults to cwd/lora_name)
            
        Returns:
            True if loaded successfully
        """
        if lora_name in self.loaded_loras:
            return True
        
        api_url = f"{self.base_url}/v1/load_lora_adapter"
        headers = {"Content-Type": "application/json"}
        
        if lora_path is None:
            lora_path = str(Path.cwd() / lora_name)
        
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        self.loaded_loras.append(lora_name)
        return True
