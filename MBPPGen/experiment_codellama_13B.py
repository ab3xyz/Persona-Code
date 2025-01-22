from transformers import AutoTokenizer
import transformers
import torch
from personality import personaGen
import json
import os
import requests
import re
from datasets import load_dataset
from personality import personaGen
from datetime import datetime

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
            max_length=5000,
        )
        return sequences[0]["generated_text"]

class experiment_codellama:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("data13B/"+self.time):
            os.makedirs("data13B/"+self.time)

    def run_compare(self, start, size):
        personas = personaGen(self.temperature)
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt=prompt[i],
                    test=testcode, code=code[i])
            code_response = codellama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open ('data13B/compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/compare_result.jsonl', 'data13B/'+ self.time +'/compare_result_'+self.time+'.jsonl')

    def run_persona(self, start, size):
        personas = personaGen(self.temperature)
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        generated_persona = []
        with open('persona.jsonl', 'r') as f:
            persona_lines = f.readlines()
            for line in persona_lines:
                persona = json.loads(line)
                print(persona["content"])
                generated_persona.append(persona["content"])
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i],testcode, persona=generated_persona[i], code=code[i])
            code_response = codellama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open ('data13B/persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/persona_result.jsonl', 'data13B/'+ self.time +'/persona_result_'+self.time+'.jsonl')

    
    def run_cot_compare(self,start, size):
        personas = personaGen(self.temperature)
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_cot_prompt(prompt=prompt[i],
                    test=testcode, code=code[i])
            code_response = codellama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open ('data13B/cot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/cot_ompare_result.jsonl', 'data13B/'+ self.time +'/cot_compare_result_'+self.time+'.jsonl')

    def run_cot_common_persona(self, start, size):
        personas = personaGen(self.temperature)
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        generated_persona = []
        with open('persona.jsonl', 'r') as f:
            persona_lines = f.readlines()
            for line in persona_lines:
                persona = json.loads(line)
                print(persona["content"])
                generated_persona.append(persona["content"])
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_cot_prompt(prompt[i],testcode, persona=generated_persona[i], code=code[i])
            code_response = codellama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open ('data13B/cot_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/cot_persona_result.jsonl', 'data13B/'+ self.time +'/cot_persona_result_'+self.time+'.jsonl')

    

    def run_few_shot_compare(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_few_shot_prompt(prompt=prompt[i], test=testcode, code=code[i], shot=shot)
            code_response = codellama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 保存代码执行结果
            file_name = '/few_shot_compare_result_'
            if shot == 3:
                file_name = '/three_shot_compare_result_'
            elif shot == 1:
                file_name = '/one_shot_compare_result_'
            with open ('data13B/few_shot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/few_shot_compare_result.jsonl', 'data13B/'+ self.time + file_name +self.time+'.jsonl')

    def run_few_shot_common_persona(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        personality = []
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        with open ("persona.jsonl", 'r') as f:
            lines = f.readlines()
            personality = [json.loads(line)["content"] for line in lines]
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_few_shot_prompt(prompt[i], testcode, persona=personality[i], code=code[i], shot=shot)
            code_response = codellama.send_request(code_prompt)
            
            # 保存代码执行结果
            file_name = '/few_shot_common_persona_result_'
            if shot == 3:
                file_name = '/three_shot_common_persona_result_'
            elif shot == 1:
                file_name = '/one_shot_common_persona_result_'
            with open ('data13B/few_shot_common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/few_shot_common_persona_result.jsonl', 'data13B/'+ self.time + file_name +self.time+'.jsonl')

    def run_shorten_persona(self, start, size):
        personas = personaGen(self.temperature)
        codellama = persona_codellama(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        generated_persona = []
        with open('shorten_persona.jsonl', 'r') as f:
            persona_lines = f.readlines()
            for line in persona_lines:
                persona = json.loads(line)
                #print(persona["content"])
                generated_persona.append(persona["content"])
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i],testcode, persona=generated_persona[i], code=code[i])
            code_response = codellama.send_request(code_prompt)

            with open ('data13B/shorten_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data13B/shorten_persona_result.jsonl', 'data13B/'+ self.time +'/shorten_persona_result_'+self.time+'.jsonl')

if __name__ == '__main__':
    exp = experiment_codellama(0.1)
    exp.run_shorten_persona(0, 427)
