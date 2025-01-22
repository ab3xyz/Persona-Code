from transformers import AutoTokenizer
import transformers
import torch
import json
import os
from datasets import load_dataset
from datetime import datetime
from personality import personaGen
from evalplus.data import get_mbpp_plus, write_jsonl




class persona_codellama:
    def __init__(self, temperature) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
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
            max_length=3000,
        )
        return sequences[0]["generated_text"]




class experiment:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        self.persona_llama = persona_codellama(temperature)
        if not os.path.exists("data_llama3_1/"+self.time):
            os.makedirs("data_llama3_1/"+self.time)
        return   
    
    def run_common_persona(self,start,size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        persona_data = []
        with open("persona.jsonl", 'r') as f:
            data = f.readlines()
            persona_data = [json.loads(persona)["content"] for persona in data]
        for i in range(start, start + size):

           
            code_prompt = personas.generate_code_prompt(prompt[i] + tests[i], persona_data[i])
            code_response = self.persona_llama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open('data_llama3_1/common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data_llama3_1/common_persona_result.jsonl', 'data_llama3_1/'+ self.time +'/common_persona_result_'+self.time+'.jsonl')

        

    def run_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_code_prompt(prompt=prompt[i])
            code_response = self.persona_llama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open('data_llama3_1/compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data_llama3_1/compare_result.jsonl', 'data_llama3_1/'+ self.time +'/compare_result_'+self.time+'.jsonl')

    def run_cot_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_cot_prompt(prompt=prompt[i])
            code_response = self.persona_llama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open('data_llama3_1/cot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data_llama3_1/cot_compare_result.jsonl', 'data_llama3_1/'+ self.time +'/cot_compare_result_'+self.time+'.jsonl')

    def run_cot_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        persona_data = []
        with open("persona.jsonl", 'r') as f:
            data = f.readlines()
            persona_data = [json.loads(persona)["content"] for persona in data]
        for i in range(start, start + size):
            code_prompt = personas.generate_cot_prompt(prompt[i] + tests[i], persona_data[i])
            code_response = self.persona_llama.send_request(code_prompt)

            with open('data_llama3_1/cot_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data_llama3_1/cot_persona_result.jsonl', 'data_llama3_1/'+ self.time +'/cot_persona_result_'+self.time+'.jsonl')

    def run_few_shot_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        for i in range(start, start + size):
            code_prompt = personas.generate_few_shot_prompt(prompt=prompt[i])
            code_response = self.persona_llama.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            with open('data_llama3_1/few_shot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data_llama3_1/few_shot_compare_result.jsonl', 'data_llama3_1/'+ self.time +'/few_shot_compare_result_'+self.time+'.jsonl')

    def run_few_shot_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, task_id, tests = personas.get_original_data()
        persona_data = []
        with open("persona.jsonl", 'r') as f:
            data = f.readlines()
            persona_data = [json.loads(persona)["content"] for persona in data]
        for i in range(start, start + size):
            code_prompt = personas.generate_few_shot_prompt(prompt[i] + tests[i], persona_data[i])
            code_response = self.persona_llama.send_request(code_prompt)

            with open('data_llama3_1/few_shot_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": code_response})
                f.write(json_str+'\n')
        os.rename('data_llama3_1/few_shot_persona_result.jsonl', 'data_llama3_1/'+ self.time +'/few_shot_persona_result_'+self.time+'.jsonl')


if __name__ == "__main__":
    ee = experiment(0.1)
    #ee.run_common_persona(0,399)
    # ee.run_cot_compare(0,399)
    # ee.run_cot_persona(0,399)
    # ee.run_few_shot_compare(0,399)
    ee.run_few_shot_persona(0,399)
