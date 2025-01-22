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
    few_shot_prompt_data = ['''
Input:
Write a function to get all Ludic numbers smaller than or equal to a given integer.

Output:
```python
def get_ludic(n):
    ludics = []
    for i in range(1, n + 1):
        ludics.append(i)
    index = 1
    while index != len(ludics):
        first_ludic = ludics[index]
        remove_index = index + first_ludic
        while remove_index < len(ludics):
            ludics.remove(ludics[remove_index])
            remove_index = remove_index + first_ludic - 1
        index += 1
    return ludics
```
''', '''
Input:  
Write a function to find the size of the largest subset of a list of numbers so that every pair is divisible.

Output:
```python
def largest_subset(a):
    n = len(a)
    dp = [0 for i in range(n)]
    dp[n - 1] = 1
    for i in range(n - 2, -1, -1):
        mxm = 0
        for j in range(i + 1, n):
            if a[j] % a[i] == 0 or a[i] % a[j] == 0:
                mxm = max(mxm, dp[j])
        dp[i] = 1 + mxm
    return max(dp)
```
''', '''
Input:
Write a Python function to find the nth Bell number.

Output:
```python
def bell_Number(n):
    bell = [[0 for i in range(n+1)] for j in range(n+1)]
    bell[0][0] = 1
    for i in range(1, n+1):
        bell[i][0] = bell[i-1][i-1]
        for j in range(1, i+1):
            bell[i][j] = bell[i-1][j-1] + bell[i][j-1]
    return bell[n][0]
```
''']



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

    def send_request(self, messages):
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
        data = get_mbpp_plus().items()
        return [problem["prompt"] for task_id, problem in data], [task_id for task_id, problem in data], [problem["assertion"] for task_id, problem in data]
    
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
                "content": "Please generate a function as a programmer with the following MBTI description:" + persona + " Don't give usage of the code, just the code."
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer, don't give usage of the code, just the code."
            })
        # prompts.append({
        #     "role": "system",
        #     "content": "The test cases are" + test + "Name the function according to the test cases but don't add them to the code."
        # })
        prompts.append({
            "role": "user",
            "content": "The problem is "+ prompt + "Don't add test cases to generated code."
        })
        print(prompt)
        return prompts
    
    def generate_cot_prompt(self, prompt, persona = ""):
        prompts = []
        if persona != "":
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer with the following MBTI description:" + persona + " Don't give usage of the code, just the code."
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer, don't give usage of the code, just the code."
            })
        prompts.append({
            "role": "system",
            "content": "You should first write a rough problem-solving process, and then output the final code. Don't give usage of the code, just the code."
        })
        prompts.append({
            "role": "user",
            "content": "The problem is "+ prompt + "Don't add test cases to generated code."
        })
        return prompts
    
    def generate_few_shot_prompt(self, prompt, persona = "", shot = 3):
        prompts = []
        if persona != "":
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer with the following MBTI description:" + persona + " Don't give usage of the code, just the code."
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer, don't give usage of the code, just the code."
            })
        for i in range(shot):
            prompts.append({
                "role": "system",
                "content": "Here is an example:\n" + self.few_shot_prompt_data[i]
            })
        prompts.append({
            "role": "user",
            "content": "The problem is "+ prompt + "Don't add test cases to generated code."
        })
        return prompts


    def parse_code(self, code):
        if "```" not in code:
            return code
        pattern = r'```python(.*?)```'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return ""
    def parse_persona(self, persona):
        pattern = r"\d+\.\s[\s\S]+"
        matches = re.findall(pattern, persona)
        return "".join(matches)