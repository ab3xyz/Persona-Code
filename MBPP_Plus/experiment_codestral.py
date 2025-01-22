import json
import os
from openai import OpenAI
import requests
import re
from datasets import load_dataset
from personality import personaGen
from datetime import datetime
from evalplus.data import get_mbpp_plus, write_jsonl

class codestral_experiment:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("MBPP_Plus/data_codestral/"+self.time):
            os.makedirs("MBPP_Plus/data_codestral/"+self.time)
        return   
    
    def run_persona(self,start,size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):

            persona_prompt = personas.generate_personality_prompt(prompt[i])
            prompt_response = personas.send_codestral_request(persona_prompt)
            # print("Persona:",personas.parse_persona(dict(prompt_response)["content"]))

            with open ('MBPP_Plus/data_codestral/persona.jsonl', 'a') as f:
                json_str = json.dumps(dict(prompt_response))
                f.write(json_str+'\n')
            code_prompt = personas.generate_code_prompt(prompt[i], dict(prompt_response)["content"])
            code_response = personas.send_codestral_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            generated_code = personas.parse_code(dict(code_response)["content"])
            print("generated code:", i, generated_code)
            
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/persona_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/persona.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/persona_'+self.time+'.jsonl')
        os.rename('MBPP_Plus/data_codestral/persona_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/persona_result_'+self.time+'.jsonl')

        

    def run_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_code_prompt(prompt=prompt[i])
            code_response = personas.send_codestral_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/compare_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/compare_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/compare_result_'+self.time+'.jsonl')


    def run_common_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        persona_data = []
        with open("MBPP_Plus/persona.jsonl", 'r') as f:
            data = f.readlines()
            persona_data = [json.loads(persona)["content"] for persona in data]
        for i in range(start, start + size):

           
            code_prompt = personas.generate_code_prompt(prompt[i], persona_data[i])
            code_response = personas.send_codestral_request(code_prompt)

            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/common_persona_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/common_persona_result_'+self.time+'.jsonl')

    def run_cot_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_cot_prompt(prompt=prompt[i])
            code_response = personas.send_codestral_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/cot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/cot_compare_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/cot_compare_result_'+self.time+'.jsonl')

    def run_cot_persona(self,start,size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        persona_data = []
        with open("MBPP_Plus/persona.jsonl", 'r') as f:
            data = f.readlines()
            persona_data = [json.loads(persona)["content"] for persona in data]
        for i in range(start, start + size):
            code_prompt = personas.generate_cot_prompt(prompt[i], persona_data[i])
            code_response = personas.send_codestral_request(code_prompt)

            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/cot_persona_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/cot_persona_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/cot_persona_result_'+self.time+'.jsonl')

    def run_few_shot_compare(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_few_shot_prompt(prompt=prompt[i], shot=shot)
            code_response = personas.send_codestral_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/few_shot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/few_shot_compare_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/few_shot_compare_result_'+self.time+'.jsonl')

    def run_few_shot_persona(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        persona_data = []
        with open("MBPP_Plus/persona.jsonl", 'r') as f:
            data = f.readlines()
            persona_data = [json.loads(persona)["content"] for persona in data]
        for i in range(start, start + size):
            code_prompt = personas.generate_few_shot_prompt(prompt[i], persona_data[i], shot)
            code_response = personas.send_codestral_request(code_prompt)

            generated_code = personas.parse_code(dict(code_response)["content"])
            # generated_code = dict(code_response)["content"]
            print("generated code: ",i, generated_code)
            result = dict(task_id=task_id[i], solution=generated_code)
            with open('MBPP_Plus/data_codestral/few_shot_persona_result.jsonl', 'a') as f:
                json_str = json.dumps(result)
                f.write(json_str+'\n')
        os.rename('MBPP_Plus/data_codestral/few_shot_persona_result.jsonl', 'MBPP_Plus/data_codestral/'+ self.time +'/few_shot_persona_result_'+self.time+'.jsonl')




if __name__ == "__main__":
    ee = codestral_experiment(0.1)
    ee.run_cot_compare(0, 399)
    ee.run_cot_persona(0, 399)
    ee.run_few_shot_compare(0, 399)
    ee.run_few_shot_persona(0, 399)