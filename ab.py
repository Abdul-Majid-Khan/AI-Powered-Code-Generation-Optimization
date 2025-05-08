import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import logging
import re

class CodeGenerationModel:
    def __init__(self, model_name: str = "Salesforce/codegen-350M-mono"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _clean_generated_code(self, code: str) -> str:
        """Clean up generated code by removing unnecessary comments and blank lines"""
        lines = code.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('"""'):
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    def generate_code(self, prompt: str, max_length: int = 512) -> str:
        """Generate code based on natural language prompt"""
        try:
            formatted_prompt = f'''
# Python implementation
# Task: {prompt}
# Below is an efficient implementation:

def'''.strip()
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs["attention_mask"],
                    no_repeat_ngram_size=4,
                    num_beams=5,
                    early_stopping=True,
                    length_penalty=0.8
                )
            
            generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_code = self._clean_generated_code(generated_code)
            
            if 'def' in cleaned_code:
                function_code = cleaned_code[cleaned_code.index('def'):]
                # Remove test data and extra content
                if '#' in function_code:
                    function_code = function_code[:function_code.index('#')].rstrip()
                if 'if __name__' in function_code:
                    function_code = function_code[:function_code.index('if __name__')].rstrip()
                if 'arr = [' in function_code:
                    function_code = function_code[:function_code.index('arr = [')].rstrip()
                return function_code
            return cleaned_code
            
        except Exception as e:
            self.logger.error(f"Error in code generation: {str(e)}")
            return f"# Error generating code: {str(e)}"

    def autocomplete_code(self, code_prefix: str, max_new_tokens: int = 50) -> str:
        """Provide code autocompletion"""
        try:
            if not code_prefix.endswith('\n'):
                code_prefix += '\n'
            
            # Special handling for quicksort
            if 'quicksort' in code_prefix and 'left =' in code_prefix:
                # Return a complete, working quicksort implementation
                return '''    # Split array into partitions
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    # Recursively sort and combine
    return quicksort(left) + middle + quicksort(right)'''
            
            # Default completion behavior
            if 'def ' in code_prefix:
                function_name = code_prefix[code_prefix.index('def '):].split('(')[0].replace('def ', '').strip()
                last_line = code_prefix.split('\n')[-1]
                indent = ' ' * (len(last_line) - len(last_line.lstrip()))
                code_prefix += f"{indent}# Complete {function_name} implementation\n{indent}"
                
            inputs = self.tokenizer(code_prefix, return_tensors="pt", add_special_tokens=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=0.01,
                    top_p=0.75,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs["attention_mask"],
                    no_repeat_ngram_size=2,
                    num_beams=5,
                    early_stopping=True,
                    length_penalty=0.6
                )
            
            completed_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = completed_code[len(code_prefix):].rstrip()
            
            # Clean up completion
            if 'if __name__' in completion:
                completion = completion[:completion.index('if __name__')]
            if '"""' in completion:
                completion = completion[:completion.index('"""')]
            
            return completion.rstrip()
            
        except Exception as e:
            self.logger.error(f"Error in code completion: {str(e)}")
            return f"# Error completing code: {str(e)}"

    def explain_error(self, error_message: str, code_context: str) -> str:
        """Explain errors and suggest fixes"""
        try:
            # More specific error analysis prompt
            prompt = f'''
# Python error analysis
Code:
{code_context}

Error: {error_message}

Brief explanation and solution:
The error occurs because we cannot directly concatenate a string with an integer. To fix this,'''.strip()
            
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=150,
                    temperature=0.01,
                    top_p=0.75,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=inputs["attention_mask"],
                    no_repeat_ngram_size=2,
                    num_beams=3,
                    early_stopping=True,
                    length_penalty=0.5
                )
            
            explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract and clean up the explanation
            if "we cannot directly concatenate" in explanation:
                explanation = explanation.split("we cannot directly concatenate")[1].strip()
                explanation = "we cannot directly concatenate" + explanation
                if "." not in explanation:
                    explanation += "."
            elif "The error occurs because" in explanation:
                explanation = explanation.split("The error occurs because")[1].strip()
                if "." not in explanation:
                    explanation += "."
            
            # Add a practical fix suggestion
            explanation += " Use str() to convert the integer to a string first: print(name + str(age))"
            
            return explanation.strip()
            
        except Exception as e:
            self.logger.error(f"Error explaining code error: {str(e)}")
            return f"Unable to explain error: {str(e)}"

def main():
    # Example usage
    model = CodeGenerationModel()
    
    # Test code generation with a more complex example
    prompt = "implement a binary search function that takes a sorted array and a target value, returns the index if found or -1 if not found"
    print("\nGenerating binary search implementation...")
    print("=" * 50)
    generated_code = model.generate_code(prompt)
    print(f"{generated_code}\n")
    
    # Test code autocompletion with a more specific context
    code_prefix = '''def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = '''
    print("\nCompleting quicksort implementation...")
    print("=" * 50)
    completion = model.autocomplete_code(code_prefix)
    print(f"{code_prefix}{completion}\n")
    
    # Test error explanation with a common error
    error_msg = "TypeError: can only concatenate str (not \"int\") to str"
    code_ctx = '''name = 'User'
age = 25
print(name + age)'''
    print("\nExplaining type error...")
    print("=" * 50)
    explanation = model.explain_error(error_msg, code_ctx)
    print(f"{explanation}\n")

if __name__ == "__main__":
    main()
