from transformers import AutoTokenizer
import transformers
import torch
import json
import os
import requests
import re
from datasets import load_dataset
from datetime import datetime
from humanEvalGen import humanEvalGen
from human_eval.data import write_jsonl, read_problems

class persona_codellama:
    def __init__(self, temperature) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="codellama/CodeLlama-13b-Instruct-hf",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.temperature = temperature

    def send_request(self, message):
        sequences = self.pipeline(
            message,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=2000,
        )
        return sequences[0]["generated_text"]
    

class experiment:
    def __init__(self, temperature) -> None:
        self.temperature = temperature
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
    def compare_method(self, prompt, position, persona_):
        heg = humanEvalGen(self.temperature)
        
        code_prompt = heg.generate_code_prompt(prompt)
        code_response = persona_.send_request(code_prompt)
        return code_response
    def common_persona_method(self, prompt, position, persona_):
        heg = humanEvalGen(self.temperature)
        
        with open("persona.jsonl", 'r') as f:
            data = f.readlines()
            personas = [json.loads(persona)["persona"] for persona in data]
            persona = personas[position]
            code_prompt = heg.generate_code_prompt(prompt, persona)
            print(code_prompt)
            code_response = persona_.send_request(code_prompt)
            return code_response
    
    def exc_experiment(self, path, method, start=0):
        problems = read_problems()
        persona_ = persona_codellama(self.temperature)
        for cnt, task_id in enumerate(problems):
            if cnt < start:
                continue
            completion=method(problems[task_id]["prompt"], cnt, persona_)
            print(completion)
            ret = dict(task_id=task_id, completion=completion)
            with open(path, 'a') as f:
                json_str = json.dumps(ret)
                f.write(json_str+'\n')


if __name__ == "__main__":
    ee = experiment(0.1)
    path = "data_13B/" + ee.time + "/common_persona.jsonl"
    if not os.path.exists("data_13B/"+ee.time):
        os.makedirs("data_13B/"+ee.time)
    ee.exc_experiment(path, method = ee.common_persona_method)
    #path2 = "data_13B/" + ee.time + "/compare.jsonl"
    #ee.exc_experiment(path2, method = ee.compare_method)
