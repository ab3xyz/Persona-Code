import json
import os
from openai import OpenAI
import requests
import re
from datasets import load_dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


class personaGen:
    prompt_data = None
    cot_prompt_data = ['''
Input:
Write a function to get all Ludic numbers smaller than or equal to a given integer.

Problem-solving process:
1. First, we need to initialize an empty list `ludics` and populate it with all integers from 1 up to the given integer `n`. This will give us a starting point to begin removing non-ludic numbers.
2. We then start with the second number in the list (index 1, as the first number is always 1). This will be our first Ludic number. After that, we will repeatedly remove every "step" number of elements, where the step is determined by the value of the current Ludic number.
3. We use a while loop to go through the list, removing non-ludic numbers. For each Ludic number, we skip and remove subsequent numbers using its value as the step. We repeat this until no more numbers can be removed.
4. Finally, the list `ludics` will contain only the Ludic numbers up to `n`. We return this list as the output.

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

Problem-solving process:
1. We are given a list of numbers. Our goal is to find the largest subset where every pair of numbers is divisible by each other. We need to approach this problem using dynamic programming. First, we initialize an array `dp` where `dp[i]` will store the size of the largest divisible subset starting from the `i`-th element.
2. We set the last element's value in `dp` to 1 because the subset with only one element has a size of 1.
3. Now, we iterate from the second last element to the first element in reverse order. For each element `a[i]`, we compare it with the elements after it (`a[j]`). If `a[j]` is divisible by `a[i]` or `a[i]` is divisible by `a[j]`, we update the maximum size of the divisible subset starting from `a[i]` by checking `dp[j]`.
4. After filling out the `dp` array, the largest divisible subset will be the maximum value in `dp`.

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

Problem-solving process:
1. The Bell number is a number that represents the number of ways to partition a set of `n` elements. To calculate the nth Bell number, we can use dynamic programming and construct a Bell triangle. We first initialize a 2D list `bell` where `bell[i][j]` represents the elements of the Bell triangle.
2. The Bell number starts with `bell[0][0] = 1`, which represents the base case, where the Bell number for a set of size 0 is 1.
3. Next, we fill out the Bell triangle using the recurrence relation:
    - The first element of each row is copied from the last element of the previous row.
    - Each subsequent element in the row is calculated by summing the element above it and the element to the left.
4. After constructing the Bell triangle, the nth Bell number is found at `bell[n][0]`.

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
    
    def send_mini4o_request(self, messages):
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=self.temperature,
        )

        return completion.choices[0].message
    
    def send_35turbo_request(self, messages):
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        print(ds.keys())
        # ds = load_dataset("google-research-datasets/mbpp", "full")
        return [ds["train"][i]["prompt"] for i in range(0, len(ds["train"]))] + [ds["test"][i]["prompt"] for i in range(0, len(ds["test"]))] + [ds["validation"][i]["prompt"] for i in range(0, len(ds["validation"]))]+ [ds["prompt"][i]["prompt"] for i in range(0, len(ds["prompt"]))], \
            [ds["train"][i]["test_list"] for i in range(0, len(ds["train"]))] + [ds["test"][i]["test_list"] for i in range(0, len(ds["test"]))] + [ds["validation"][i]["test_list"] for i in range(0, len(ds["validation"]))]+ [ds["prompt"][i]["test_list"] for i in range(0, len(ds["prompt"]))]
    
    def get_original_data_with_code(self):
        ds = load_dataset("google-research-datasets/mbpp", "sanitized")
        # ds = load_dataset("google-research-datasets/mbpp", "full")
        return [ds["train"][i]["prompt"] for i in range(0, len(ds["train"]))] + [ds["test"][i]["prompt"] for i in range(0, len(ds["test"]))] + [ds["validation"][i]["prompt"] for i in range(0, len(ds["validation"]))] + [ds["prompt"][i]["prompt"] for i in range(0, len(ds["prompt"]))], \
            [ds["train"][i]["test_list"] for i in range(0, len(ds["train"]))] + [ds["test"][i]["test_list"] for i in range(0, len(ds["test"]))] + [ds["validation"][i]["test_list"] for i in range(0, len(ds["validation"]))]+ [ds["prompt"][i]["test_list"] for i in range(0, len(ds["prompt"]))], \
            [ds["train"][i]["code"] for i in range(0, len(ds["train"]))] + [ds["test"][i]["code"] for i in range(0, len(ds["test"]))] + [ds["validation"][i]["code"] for i in range(0, len(ds["validation"]))]+ [ds["prompt"][i]["code"] for i in range(0, len(ds["prompt"]))]

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
    
    def generate_code_prompt(self, prompt, test, persona = "", code = ""):
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
        if code != "":
            prompts.append({
                "role": "user",
                "content": "The function name should be " + self.capture_function_name(json.dumps(code))+", The problem is " + prompt
            })
        else:
            prompts.append({
                "role": "user",
                "content": prompt
            })
        print(self.capture_function_name(json.dumps(code)))
        print(prompt)
        return prompts
    
    def generate_cot_prompt(self, prompt, test,  persona = "", code = "", shot = 0):
        prompts = []
        if persona != "":
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer with the following MBTI description:" + persona
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer."
            })
        prompts.append({
            "role": "system",
            "content": "You should first write a rough problem-solving process, and then output the final code. Don't give usage of the code, just the code."
        })
        for i in range(min(shot, len(self.cot_prompt_data))):
            prompts.append({
                "role": "system",
                "content": "Here is an example:\n" + self.cot_prompt_data[i]
            })
        if code != "":
            prompts.append({
                "role": "user",
                "content": "The function name should be " + self.capture_function_name(json.dumps(code))+", The problem is " + prompt + " Lets think step by step."
            })
        else:
            prompts.append({
                "role": "user",
                "content": prompt + " Lets think step by step."
            })
        # print(self.capture_function_name(json.dumps(code)))
        # print(prompt)
        return prompts
    
    def generate_few_shot_prompt(self, prompt, test,  persona = "", code = "", shot = 0):
        prompts = []
        if persona != "":
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer with the following MBTI description:" + persona
            })
        else:
            prompts.append({
                "role": "system",
                "content": "Please generate a function as a programmer."
            })
        for i in range(min(shot, len(self.few_shot_prompt_data))):
            prompts.append({
                "role": "system",
                "content": "Here is an example:\n" + self.few_shot_prompt_data[i]
            })
        if code != "":
            prompts.append({
                "role": "user",
                "content": "The function name should be " + self.capture_function_name(json.dumps(code))+", The problem is " + prompt
            })
        else:
            prompts.append({
                "role": "user",
                "content": prompt
            })
        print(prompts)
        return prompts


    def parse_code(self, code):
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