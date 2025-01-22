from human_eval.data import write_jsonl, read_problems
import json
from openai import OpenAI
import re
from execute import CodeExecutor
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

class humanEvalGen:
    def __init__(self, temperature) -> None:
        self.temperature = temperature
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        self.MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
        self.DEEP_SEEK_API_KEY = os.environ.get("DEEP_SEEK_API_KEY")
        self.QWEN_API_KEY = os.environ.get("QWEN_API_KEY")
        return
    
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
    
    def send_codestral_request(self, messages):
        client = MistralClient(api_key=self.MISTRAL_API_KEY)
        chat_response = client.chat(
            model="codestral-latest",
            messages=[ChatMessage(role = message["role"], content = message["content"]) for message in messages],
        )
        return chat_response.choices[0].message

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
    
    def parse_code(self, code):
        pattern = r'```python(.*?)```'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            if "```" in code:
                new_pattern = r'```(.*?)```'
                new_match = re.search(new_pattern, code, re.DOTALL)
                if new_match:
                    return new_match.group(1).strip()
            return code
        
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
                "content": "Please complete the function as a programmer with the following MBTI description:" + persona + " Don't give usage of the code, just the code."
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please complete the function as a programmer, don't give usage of the code, just the code."
            })
        prompts.append({
            "role": "user",
            "content": prompt
        })
        return prompts
    
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


    def experiment(self, method, path, start = 0):
        problems = read_problems()
        num_samples_per_task = 1
        exe = CodeExecutor()
        cnt = start
        for task_id in problems:
            if cnt > 0:
                cnt -= 1
                continue
            completion=method(problems[task_id]["prompt"])
            print(completion)
            check_program = completion + "\n" + problems[task_id]["test"] + "\n" + f"check({problems[task_id]['entry_point']})"
            result = exe.execute_code(check_program)
            print(result)
            ret = dict(task_id=task_id, completion=completion, result=str(result), success = exe.check_result(result))
            with open(path, 'a') as f:
                json_str = json.dumps(ret)
                f.write(json_str+'\n')

    def new_experiment(self, method, path, start = 0):
        problems = read_problems()
        num_samples_per_task = 1
        exe = CodeExecutor()
        for cnt, task_id in enumerate(problems):
            if cnt < start:
                continue
            print("="*10)
            completion=method(problems[task_id]["prompt"], cnt)
            print(completion)
            prefix = "from typing import List\nimport math\n"
            check_program = prefix + completion + "\n" + problems[task_id]["test"] + "\n" + f"check({problems[task_id]['entry_point']})"
            result = exe.execute_code(check_program)
            print(result)
            ret = dict(task_id=task_id, completion=completion, result=str(result), success = exe.check_result(result))
            with open(path, 'a') as f:
                json_str = json.dumps(ret)
                f.write(json_str+'\n')
