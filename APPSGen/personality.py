import json
import os
from openai import OpenAI
import requests
import re
from datasets import load_dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from evalplus.data import get_mbpp_plus, write_jsonl


class personaGen:
    prompt_data = None
    def __init__(self, temperature) -> None:
        self.prompt_data = self.get_original_data()     
        self.temperature = temperature
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        self.MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
        self.DEEP_SEEK_API_KEY = os.environ.get("DEEP_SEEK_API_KEY")
        self.QWEN_API_KEY = os.environ.get("QWEN_API_KEY")
        return
    def check_python_code(self, code):
        try:
            compile(code, '<string>', 'exec')
            print("Code is valid and can be compiled.")
            return True
        except SyntaxError as e:
            print(f"SyntaxError: {e}")
            print("Code has syntax errors and cannot be compiled.")
            return False

    def send_4o_request(self, messages):
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=self.temperature,
        )

        return completion.choices[0].message
    
    def send_4omini_request(self, messages):
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=self.temperature,
        )

        return completion.choices[0].message
    
    def send_deepseek_request(self, messages, type = "coder"):
        client = OpenAI(api_key=self.DEEP_SEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
        model="deepseek-coder" if type == "coder" else "deepseek-chat",
        messages=messages,
        temperature=self.temperature
        )
        return response.choices[0].message


    def send_qwen_request(self, messages):
        client = OpenAI(api_key=self.QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        response = client.chat.completions.create(
        model="qwen-long",
        messages=messages,
        temperature=self.temperature
        )
        return response.choices[0].message
    
    def send_codestral_request(self, messages):
        client = MistralClient(api_key=self.MISTRAL_API_KEY)
        chat_response = client.chat(
            model="codestral-latest",
            messages=[ChatMessage(role = message["role"], content = message["content"]) for message in messages],
        )
        return chat_response.choices[0].message

    def get_original_data(self):
        ds = load_dataset("codeparrot/apps", "interview",trust_remote_code=True)
        return [ds["test"][i]["question"] for i in range(500)], [ds["test"][i]["input_output"] for i in range(500)]
    
    def generate_personality_prompt(self, prompt):
        prompts = []
        prompts.append({
            "role": "system",
            "content": "Please generate the MBTI of the programmer, \
                of whom can best answer this question. \
                Please provide a detailed description of the MBTI. \
                Don't give a direct answer to the question."
        })
        prompts.append({
            "role": "user",
            "content": prompt
        })
        return prompts
    
    def capture_function_name(self, code):
        # pattern = re.compile(r'def\s+(\w+)\s*\(.*\)\s*:')
        pattern = re.compile(r'def\s+\w+\s*\(.*?\)\s*:') 
        # 查找函数定义
        match = pattern.search(code)
        if match:
            return match.group(0)
        else:
            return ""

    def generate_realworld_problem_prompt(self, prompt):
        prompts = []
        prompts.append({
            "role": "system",
            "content": "Please generate a real-world problem base on the following description. Please give a concise description of only the problem."
        })
        prompts.append({
            "role": "user",
            "content": prompt
        })
        return prompts
    
    def generate_identity_prompt(self, prompt):
        prompts = []
        prompts.append({
            "role": "system",
            "content": "Please generate a person who may encounter this real-world problem. Please give a concise description."
        })
        prompts.append({
            "role": "user",
            "content": prompt
        })
        return prompts
    
    def generate_persona_on_identities(self, prompt):
        prompts = []
        prompts.append({
            "role": "system",
            "content": "Please generate the possible MBTI of the person based on the description. Please just give the four characters."
        })
        prompts.append({
            "role": "user",
            "content": prompt
        })
        return prompts
    
    def generate_code_prompt(self, prompt, persona = ""):
        prompts = []
        if persona != "":
            prompts.append({
                "role": "system",
                "content": "Please generate the solution python code as a programmer with the following MBTI description:" + persona + " Don't give usage of the code, just the code."
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please generate the solution python code as a programmer, don't give usage of the code, just the code."
            })
        # prompts.append({
        #     "role": "system",
        #     "content": "The test cases are" + test + "Name the function according to the test cases but don't add them to the code."
        # })
        prompts.append({
            "role": "user",
            "content": "The problem is "+ prompt + "Don't add test cases to generated code. You need to consider the input and print the output."
        })
        print(prompt)
        return prompts

    def parse_code(self, code):
        if "```" not in code:
            return code
        pattern = r'```python(.*?)```'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            new_pattern = r'```(.*?)```'
            new_match = re.search(new_pattern, code, re.DOTALL)
            if new_match:
                return new_match.group(1).strip()
            else:
                return ""
    def parse_persona(self, persona):
        pattern = r"\d+\.\s[\s\S]+"
        matches = re.findall(pattern, persona)
        return "".join(matches)