"""LLM integration for code generation and modification."""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import config


@dataclass
class CodeChange:
    """Represents a code change to be made."""
    file_path: str
    action: str  # 'create', 'modify', 'delete'
    content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    description: str = ""


class LLMProvider:
    """Base class for LLM providers."""
    
    def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code based on prompt and context."""
        raise NotImplementedError
    
    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code and answer questions about it."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT integration."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        
    def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code using OpenAI GPT."""
        if not self.api_key:
            return self._fallback_response(prompt)
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            system_prompt = """You are an expert programmer. Generate clean, well-documented code.
Focus on:
- Writing readable, maintainable code
- Adding appropriate comments
- Following best practices
- Handling edge cases
- Including error handling where appropriate

Return only the code, no explanations unless specifically asked."""

            if context:
                system_prompt += f"\n\nContext about the codebase:\n{context}"
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 2000,
                'temperature': 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"OpenAI API error: {response.status_code}")
                return self._fallback_response(prompt)
                
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._fallback_response(prompt)
    
    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code using OpenAI GPT."""
        if not self.api_key:
            return "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""Analyze this code and answer the question:

Code:
```
{code}
```

Question: {question}

Provide a clear, concise answer."""
            
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'system', 'content': 'You are an expert code analyst. Provide clear, accurate analysis of code.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 1000,
                'temperature': 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"API error: {response.status_code}"
                
        except Exception as e:
            return f"Error analyzing code: {e}"
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when API is not available."""
        return f"""# Generated code for: {prompt}
# Note: This is a placeholder. Configure OpenAI API key for actual code generation.

def placeholder_function():
    \"\"\"
    This is a placeholder function.
    To get actual code generation, please:
    1. Set your OPENAI_API_KEY environment variable
    2. Run the command again
    \"\"\"
    pass
"""


class OllamaProvider(LLMProvider):
    """Local Ollama integration for privacy-focused users."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "codellama"):
        self.base_url = base_url
        self.model = model
    
    def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code using local Ollama."""
        try:
            full_prompt = f"""You are an expert programmer. Generate clean, well-documented code.

{f"Context: {context}" if context else ""}

Request: {prompt}

Generate only the code, no explanations:"""

            data = {
                'model': self.model,
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'top_p': 0.9
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return self._fallback_local_response(prompt)
                
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._fallback_local_response(prompt)
    
    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code using local Ollama."""
        try:
            prompt = f"""Analyze this code and answer the question:

Code:
{code}

Question: {question}

Answer:"""
            
            data = {
                'model': self.model,
                'prompt': prompt,
                'stream': False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return "Ollama service not available"
                
        except Exception as e:
            return f"Error with Ollama: {e}"
    
    def _fallback_local_response(self, prompt: str) -> str:
        """Fallback when Ollama is not available."""
        return f"""# Generated code for: {prompt}
# Note: Ollama not available. Install Ollama and run 'ollama pull codellama' for local code generation.

def placeholder_function():
    \"\"\"
    This is a placeholder function.
    To get local code generation:
    1. Install Ollama (https://ollama.ai)
    2. Run: ollama pull codellama
    3. Start Ollama service
    \"\"\"
    pass
"""


class LLMManager:
    """Manages LLM providers and code generation."""
    
    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(),
            'ollama': OllamaProvider()
        }
        self.default_provider = 'openai'
    
    def set_provider(self, provider_name: str):
        """Set the active LLM provider."""
        if provider_name in self.providers:
            self.default_provider = provider_name
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def generate_code(self, prompt: str, context: str = "", provider: str = None) -> str:
        """Generate code using the specified or default provider."""
        provider_name = provider or self.default_provider
        provider_obj = self.providers[provider_name]
        return provider_obj.generate_code(prompt, context)
    
    def analyze_code(self, code: str, question: str, provider: str = None) -> str:
        """Analyze code using the specified or default provider."""
        provider_name = provider or self.default_provider
        provider_obj = self.providers[provider_name]
        return provider_obj.analyze_code(code, question)
    
    def plan_changes(self, request: str, codebase_context: str) -> List[CodeChange]:
        """Plan what changes need to be made to fulfill a request."""
        prompt = f"""Given this request and codebase context, plan the specific changes needed.

Request: {request}

Codebase context:
{codebase_context}

Return a JSON list of changes in this format:
[
  {{
    "file_path": "path/to/file.py",
    "action": "create|modify|delete",
    "description": "What this change does",
    "content": "The new/modified code content"
  }}
]

Focus on minimal, targeted changes. Only modify what's necessary."""

        response = self.generate_code(prompt)
        
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                changes_data = json.loads(json_match.group())
                changes = []
                for change_data in changes_data:
                    changes.append(CodeChange(
                        file_path=change_data['file_path'],
                        action=change_data['action'],
                        content=change_data.get('content', ''),
                        description=change_data.get('description', '')
                    ))
                return changes
        except Exception as e:
            print(f"Error parsing change plan: {e}")
        
        # Fallback: create a single change
        return [CodeChange(
            file_path="generated_code.py",
            action="create",
            content=response,
            description=f"Generated code for: {request}"
        )]
