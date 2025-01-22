import json
import os
from openai import OpenAI
import requests
import re
from datasets import load_dataset
from personality import personaGen
from datetime import datetime

class GPT4o_experiment:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("APPSGen/data_4o/"+self.time):
            os.makedirs("APPSGen/data_4o/"+self.time)
        return   
    
    def run_persona(self,start,size):
        personas = personaGen(self.temperature)
        prompt, tests = personas.get_original_data()
        for i in range(start, start + size):
            # 产生人格的prompt
            persona_prompt = personas.generate_personality_prompt(prompt[i])
            prompt_response = personas.send_4o_request(persona_prompt)
            # print("Persona:",personas.parse_persona(dict(prompt_response)["content"]))
            # 产生代码的prompt
            with open ('APPSGen/data_4o/persona.jsonl', 'a') as f:
                json_str = json.dumps(dict(prompt_response))
                f.write(json_str+'\n')
            code_prompt = personas.generate_code_prompt(prompt[i], dict(prompt_response)["content"])
            code_response = personas.send_4o_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            generated_code = personas.parse_code(dict(code_response)["content"])
            print("generated code:", i, generated_code)
            
            result = dict(solution=generated_code)
            with open('APPSGen/data_4o/persona_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('APPSGen/data_4o/persona.jsonl', 'APPSGen/data_4o/'+ self.time +'/persona_'+self.time+'.jsonl')
        os.rename('APPSGen/data_4o/persona_result.jsonl', 'APPSGen/data_4o/'+ self.time +'/persona_result_'+self.time+'.jsonl')

        

    def run_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_code_prompt(prompt=prompt[i])
            code_response = personas.send_4o_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            
            result = dict(solution=generated_code)
            with open('APPSGen/data_4o/compare_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('APPSGen/data_4o/compare_result.jsonl', 'APPSGen/data_4o/'+ self.time +'/compare_result_'+self.time+'.jsonl')
    

if __name__ == "__main__":
    ee = GPT4o_experiment(0.1)
    # ee.run_persona(0,500)
    ee.run_compare(0,500)
    ee = GPT4o_experiment(0.1)
    # ee.run_persona(0,500)
    ee.run_compare(0,500)
