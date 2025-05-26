"""LLM integration for code generation and modification."""

import os
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import core.config as config

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except (ImportError, UnicodeDecodeError):
    # If dotenv fails, try to manually load from .env file
    try:
        with open('.env', 'r', encoding='utf-8-sig') as f:  # Handle BOM
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except Exception:
        pass  # Use system env vars if all else fails


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

        # Only show API key status if explicitly requested or if this provider is being used
        # The warning will be shown when actually trying to use OpenAI

    def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code using OpenAI GPT."""
        if not self.api_key:
            print("❌ No API key, using fallback response")
            return self._fallback_response(prompt)

        print(f"🔄 Making OpenAI API call for prompt: {prompt[:50]}...")
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            system_prompt = """You are an expert programmer. Generate COMPLETE, WORKING code that runs immediately without any placeholders, undefined functions, or missing imports.

CRITICAL REQUIREMENTS:
- Write COMPLETE, SELF-CONTAINED code that runs immediately
- NO placeholders like "load_data()" or "undefined_function()"
- NO missing imports - include ALL necessary imports at the top
- NO TODO comments or incomplete functions
- DEFINE ALL functions and classes that are used
- Include ALL data generation, processing, and output in one file
- The code must execute successfully from start to finish
- If you need data, GENERATE it in the code (don't load from external files)
- If you need libraries, use only standard library or generate alternatives
- Include proper error handling and docstrings
- Add a main section that demonstrates the functionality

Return ONLY the complete code that runs without errors, no explanations."""

            if context:
                system_prompt += f"\n\nContext about the codebase:\n{context}"

            data = {
                'model': 'gpt-3.5-turbo',  # Use GPT-3.5-turbo (more widely available)
                'messages': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 3000,  # Increased for more complete code
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
                generated_code = result['choices'][0]['message']['content'].strip()
                print(f"✅ OpenAI API success! Generated {len(generated_code)} characters")
                return generated_code
            else:
                print(f"❌ OpenAI API error: {response.status_code} - {response.text}")
                return self._fallback_response(prompt)

        except Exception as e:
            print(f"❌ Error calling OpenAI API: {e}")
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
        # Generate actual code based on common patterns
        if "factorial" in prompt.lower():
            return """def factorial(n):
    \"\"\"Calculate the factorial of a number.\"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

# Example usage
if __name__ == "__main__":
    print(f"Factorial of 5: {factorial(5)}")
"""
        elif "fibonacci" in prompt.lower():
            return """def fibonacci(n):
    \"\"\"Generate the nth Fibonacci number.\"\"\"
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def fibonacci_sequence(count):
    \"\"\"Generate a sequence of Fibonacci numbers.\"\"\"
    return [fibonacci(i) for i in range(count)]

# Example usage
if __name__ == "__main__":
    print(f"Fibonacci(10): {fibonacci(10)}")
    print(f"First 10 Fibonacci numbers: {fibonacci_sequence(10)}")
"""
        elif "sort" in prompt.lower():
            return """def bubble_sort(arr):
    \"\"\"Sort an array using bubble sort algorithm.\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def quick_sort(arr):
    \"\"\"Sort an array using quick sort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_array}")
    print(f"Bubble sorted: {bubble_sort(test_array.copy())}")
    print(f"Quick sorted: {quick_sort(test_array.copy())}")
"""
        elif "calculator" in prompt.lower() or "math" in prompt.lower():
            return """class Calculator:
    \"\"\"A simple calculator class with basic operations.\"\"\"

    def add(self, a, b):
        \"\"\"Add two numbers.\"\"\"
        return a + b

    def subtract(self, a, b):
        \"\"\"Subtract b from a.\"\"\"
        return a - b

    def multiply(self, a, b):
        \"\"\"Multiply two numbers.\"\"\"
        return a * b

    def divide(self, a, b):
        \"\"\"Divide a by b.\"\"\"
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def power(self, base, exponent):
        \"\"\"Calculate base raised to the power of exponent.\"\"\"
        return base ** exponent

# Example usage
if __name__ == "__main__":
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"6 * 7 = {calc.multiply(6, 7)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    print(f"2^8 = {calc.power(2, 8)}")
"""
        else:
            # Generic function template
            function_name = prompt.lower().replace(" ", "_").replace("create", "").replace("function", "").replace("to", "").strip()
            if not function_name:
                function_name = "custom_function"

            return f"""def {function_name}():
    \"\"\"
    {prompt}

    This function was generated based on your request.
    Please modify it according to your specific needs.
    \"\"\"
    # TODO: Implement the logic for: {prompt}
    pass

# Example usage
if __name__ == "__main__":
    result = {function_name}()
    print(f"Result: {{result}}")
"""


class OllamaProvider(LLMProvider):
    """Local Ollama integration for privacy-focused users."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = None):
        self.base_url = base_url
        # Try more powerful models in order of preference
        self.preferred_models = [
            "qwen2.5-coder:7b",      # Most powerful coding model
            "deepseek-coder:6.7b",   # Good coding model
            "codellama:latest",      # Reliable coding model
            "deepseek-r1:7b"         # Reasoning model (last - has thinking issues)
        ]
        self.model = self._select_best_model(model)
        print(f"🚀 Using enhanced model: {self.model}")

    def _select_best_model(self, requested_model: str = None) -> str:
        """Select the best available model."""
        if requested_model:
            return requested_model

        # Check which models are available
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            available_models = result.stdout

            for model in self.preferred_models:
                if model in available_models:
                    return model

        except Exception:
            pass

        return "codellama:latest"  # Default fallback

    def generate_code(self, prompt: str, context: str = "") -> str:
        """Generate code using local Ollama."""
        try:
            # CodeLlama works better with specific formatting
            full_prompt = f"""<s>[INST] You are an expert Python programmer. Generate COMPLETE, WORKING Python code that runs immediately without any placeholders, undefined functions, or missing imports.

Request: {prompt}

{f"Context from codebase: {context}" if context else ""}

CRITICAL REQUIREMENTS:
- Write COMPLETE, SELF-CONTAINED Python code that runs immediately
- NO placeholders like "load_data()" or "undefined_function()"
- NO missing imports - include ALL necessary imports at the top
- NO TODO comments or incomplete functions
- DEFINE ALL functions and classes that are used
- Include ALL data generation, processing, and output in one file
- The code must execute successfully from start to finish
- If you need data, GENERATE it in the code (don't load from external files)

LIBRARY RESTRICTIONS - ONLY USE PYTHON STANDARD LIBRARY:
- NO numpy, pandas, sklearn, tensorflow, keras, torch, pygame, matplotlib, scipy
- NO external libraries whatsoever
- Use ONLY: math, random, json, csv, os, sys, time, datetime, collections, itertools, etc.
- For neural networks: implement matrix operations manually with lists and loops
- For data processing: use basic Python data structures (lists, dicts)
- For math: use math module and basic arithmetic

EXAMPLE: If asked for "neural network", implement:
- Matrix operations using nested lists
- Activation functions using math.exp, math.tanh
- Training loop with basic Python loops
- Synthetic data generation using random module
- ALL in one complete file using only standard library

Generate ONLY the complete Python code that runs without errors: [/INST]

```python"""

            data = {
                'model': self.model,
                'prompt': full_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.2,        # Slightly higher for creativity
                    'top_p': 0.95,            # Higher for better diversity
                    'top_k': 50,              # Control vocabulary selection
                    'num_predict': 4096,      # Much longer responses
                    'repeat_penalty': 1.1,    # Prevent repetition
                    'num_ctx': 8192,          # Larger context window
                    'num_thread': 8,          # Use more CPU threads
                    # Remove stop tokens that cause premature truncation
                    # 'stop': ['```', '</s>']  # Commented out to prevent truncation
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                code = result['response'].strip()

                # Enhanced response cleaning for better code extraction
                # First, remove reasoning tags from deepseek-r1 model
                if '<think>' in code:
                    # Remove everything from <think> to </think>
                    import re
                    code = re.sub(r'<think>.*?</think>', '', code, flags=re.DOTALL)

                # Remove markdown formatting
                if code.startswith('```python'):
                    code = code[9:]
                elif code.startswith('```'):
                    code = code[3:]

                # Remove trailing markdown
                if code.endswith('```'):
                    code = code[:-3]

                # Remove any remaining artifacts
                code = code.replace('```python', '').replace('```', '')

                # Split into lines for advanced cleaning
                lines = code.split('\n')
                cleaned_lines = []
                in_code_block = True

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Skip empty lines at the start
                    if not cleaned_lines and not stripped:
                        continue

                    # Detect end of code (explanatory text or reasoning)
                    if (stripped.startswith('This code') or
                        stripped.startswith('The code') or
                        stripped.startswith('Here') or
                        stripped.startswith('Note:') or
                        stripped.startswith('Explanation:') or
                        stripped.startswith('<think>') or
                        stripped.startswith('</think>') or
                        'think>' in stripped):
                        break

                    # Skip incomplete assignments
                    if stripped.endswith('=') and '=' in stripped:
                        continue

                    # Skip incomplete function calls
                    if stripped.endswith('(') and not stripped.startswith('#'):
                        continue

                    # Fix common syntax issues
                    if stripped.startswith('sorted_files =') and stripped.endswith('='):
                        continue  # Skip malformed assignments

                    cleaned_lines.append(line)

                # Join and final cleanup
                code = '\n'.join(cleaned_lines).strip()

                # Validate basic Python syntax
                if not self._is_valid_python_structure(code):
                    print("⚠️  Generated code has structural issues, applying fixes...")
                    code = self._fix_common_issues(code)

                return code
            else:
                print(f"Ollama API error: {response.status_code}")
                return self._fallback_local_response(prompt)

        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return self._fallback_local_response(prompt)

    def _is_valid_python_structure(self, code: str) -> bool:
        """Check if code has basic Python structure."""
        try:
            # Basic checks for Python structure
            lines = code.split('\n')
            has_imports_or_functions = any(
                line.strip().startswith(('import ', 'from ', 'def ', 'class '))
                for line in lines
            )

            # Check for incomplete assignments
            has_incomplete_assignments = any(
                line.strip().endswith('=') and '=' in line.strip()
                for line in lines
            )

            return has_imports_or_functions and not has_incomplete_assignments
        except:
            return False

    def _fix_common_issues(self, code: str) -> str:
        """Fix common code generation issues."""
        lines = code.split('\n')
        fixed_lines = []

        for line in lines:
            stripped = line.strip()

            # Fix incomplete assignments
            if stripped.endswith('=') and '=' in stripped:
                if 'passwords' in stripped:
                    fixed_lines.append(line.replace('passwords =', 'passwords = []'))
                elif 'files' in stripped:
                    fixed_lines.append(line.replace('files =', 'files = []'))
                elif 'data' in stripped:
                    fixed_lines.append(line.replace('data =', 'data = []'))
                else:
                    continue  # Skip if we can't fix it
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def analyze_code(self, code: str, question: str) -> str:
        """Analyze code using local Ollama."""
        try:
            prompt = f"""<s>[INST] You are an expert code analyst. Analyze the following Python code and answer the question.

Code:
```python
{code}
```

Question: {question}

Provide a clear, detailed analysis: [/INST]"""

            data = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'top_p': 0.9
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=90
            )

            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return f"Ollama service error: {response.status_code}"

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
        self.default_provider = 'ollama'

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

        # First, analyze what files exist in the context
        existing_files = []
        if codebase_context:
            # Extract file information from context
            lines = codebase_context.split('\n')
            for line in lines:
                if line.startswith('File:'):
                    file_name = line.replace('File:', '').strip()
                    if file_name and file_name != 'unknown':
                        existing_files.append(file_name)

        # Generate the actual code for the request
        code_prompt = f"""Generate Python code for this request: {request}

{f"Existing codebase context: {codebase_context}" if codebase_context else ""}

Generate complete, working Python code with proper error handling and documentation."""

        generated_code = self.generate_code(code_prompt)

        # Determine the best file to modify or create
        if existing_files:
            # If we have existing files, try to modify the most relevant one
            target_file = existing_files[0]  # Use the first relevant file
            action = "modify"
            description = f"Add functionality: {request}"
        else:
            # Create a new file
            if "calculator" in request.lower() or "math" in request.lower():
                target_file = "calculator.py"
            elif "test" in request.lower():
                target_file = "test_functions.py"
            elif "util" in request.lower():
                target_file = "utilities.py"
            else:
                target_file = "new_feature.py"
            action = "create"
            description = f"Create new file for: {request}"

        return [CodeChange(
            file_path=target_file,
            action=action,
            content=generated_code,
            description=description
        )]
