import json
import os
import requests
import re
from datasets import load_dataset
from execute import CodeExecutor
from personality import personaGen
from datetime import datetime

class experiment:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("MBPPGen/data/"+self.time):
            os.makedirs("MBPPGen/data/"+self.time)
        return   
    
    def run_identity_to_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, test = personas.get_original_data()
        for i in range(start, start+size):
            # 产生人格的prompt
            persona_log = {}
            realworld_prompt = personas.generate_realworld_problem_prompt(prompt[i])
            prompt_response = personas.send_request(realworld_prompt)
            persona_log["realworld"] = dict(prompt_response)["content"]
            identity_prompt = personas.generate_identity_prompt(persona_log["realworld"])
            prompt_response = personas.send_request(identity_prompt)
            persona_log["identity"] = dict(prompt_response)["content"]
            persona_prompt = personas.generate_persona_on_identities(persona_log["identity"])
            prompt_response = personas.send_request(persona_prompt)
            persona_log["persona"] = dict(prompt_response)["content"]

            # print("Persona:",personas.parse_persona(dict(prompt_response)["content"]))
            # 产生代码的prompt
            
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, persona_log['persona'])
            code_response = personas.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))


            with open ('MBPPGen/data/identity_persona.jsonl', 'a') as f:
                json_str = json.dumps(dict(persona_log))
                f.write(json_str+'\n')
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)

            # 保存代码执行结果
            with open ('MBPPGen/data/identity_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/identity_persona.jsonl', 'MBPPGen/data/'+ self.time +'/identity_persona_'+self.time+'.jsonl')
        os.rename('MBPPGen/data/identity_persona_result.jsonl', 'MBPPGen/data/'+ self.time +'/identity_persona_result_'+self.time+'.jsonl')
    
    
    
    
    def run_persona(self,start,size):
        personas = personaGen(self.temperature)
        prompt, test = personas.get_original_data()
        for i in range(start, start + size):
            # 产生人格的prompt
            persona_prompt = personas.generate_personality_prompt(prompt[i])
            prompt_response = personas.send_request(persona_prompt)
            # print("Persona:",personas.parse_persona(dict(prompt_response)["content"]))
            # 产生代码的prompt
            with open ('MBPPGen/data/persona.jsonl', 'a') as f:
                json_str = json.dumps(dict(prompt_response))
                f.write(json_str+'\n')
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, dict(prompt_response)["content"])
            code_response = personas.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')

            # 保存代码执行结果
            with open ('MBPPGen/data/persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/persona.jsonl', 'MBPPGen/data/'+ self.time +'/persona_'+self.time+'.jsonl')
        os.rename('MBPPGen/data/persona_result.jsonl', 'MBPPGen/data/'+ self.time +'/persona_result_'+self.time+'.jsonl')

    def run_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, code=code[i])
            code_response = personas.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')

            # 保存代码执行结果
            with open ('MBPPGen/data/compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/compare_result.jsonl', 'MBPPGen/data/'+ self.time +'/compare_result_'+self.time+'.jsonl')

    def run_common_persona(self, start, size):
        personas = personaGen(self.temperature)
        personality = []
        prompt, test, code = personas.get_original_data_with_code()
        with open ("MBPPGen/persona.jsonl", 'r') as f:
            lines = f.readlines()
            personality = [json.loads(line)["content"] for line in lines]
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt=prompt[i], test=testcode, persona=personality[i] , code=code[i])
            code_response = personas.send_request(code_prompt)
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')

            # 保存代码执行结果
            with open ('MBPPGen/data/common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/common_persona_result.jsonl', 'MBPPGen/data/'+ self.time +'/common_persona_result_'+self.time+'.jsonl')



    def run_cot_compare(self,start, size):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_cot_prompt(prompt[i], testcode, code=code[i])
            code_response = personas.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')

            # 保存代码执行结果
            with open ('MBPPGen/data/cot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/cot_compare_result.jsonl', 'MBPPGen/data/'+ self.time +'/cot_compare_result_'+self.time+'.jsonl')

    def run_cot_common_persona(self, start, size):
        personas = personaGen(self.temperature)
        personality = []
        prompt, test, code = personas.get_original_data_with_code()
        with open ("MBPPGen/persona.jsonl", 'r') as f:
            lines = f.readlines()
            personality = [json.loads(line)["content"] for line in lines]
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_cot_prompt(prompt=prompt[i], test=testcode, persona=personality[i], code=code[i])
            code_response = personas.send_request(code_prompt)
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)

            # 保存代码执行结果
            with open ('MBPPGen/data/cot_common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/cot_common_persona_result.jsonl', 'MBPPGen/data/'+ self.time +'/cot_common_persona_result_'+self.time+'.jsonl')

    def run_few_shot_compare(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_few_shot_prompt(prompt=prompt[i], test=testcode, code=code[i], shot=shot)
            code_response = personas.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')

            # 保存代码执行结果
            file_name = '/few_shot_compare_result_'
            if shot == 3:
                file_name = '/three_shot_compare_result_'
            elif shot == 1:
                file_name = '/one_shot_compare_result_'
            with open ('MBPPGen/data/few_shot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/few_shot_compare_result.jsonl', 'MBPPGen/data/'+ self.time + file_name +self.time+'.jsonl')

    def run_few_shot_common_persona(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        personality = []
        prompt, test, code = personas.get_original_data_with_code()
        with open ("MBPPGen/persona.jsonl", 'r') as f:
            lines = f.readlines()
            personality = [json.loads(line)["content"] for line in lines]
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_few_shot_prompt(prompt[i], testcode, persona=personality[i], code=code[i], shot=shot)
            code_response = personas.send_request(code_prompt)
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)

            # 保存代码执行结果
            file_name = '/few_shot_common_persona_result_'
            if shot == 3:
                file_name = '/three_shot_common_persona_result_'
            elif shot == 1:
                file_name = '/one_shot_common_persona_result_'
            with open ('MBPPGen/data/few_shot_common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/few_shot_common_persona_result.jsonl', 'MBPPGen/data/'+ self.time + file_name +self.time+'.jsonl')

    def run_shorten_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, test = personas.get_original_data()
        with open ("MBPPGen/persona.jsonl", 'r') as f:
            lines = f.readlines()
            personality = [json.loads(line)["content"] for line in lines]
        for i in range(start, start+size):
            # 产生代码的prompt
            
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, personality[i])
            code_response = personas.send_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            
            # 执行代码并进行检查
            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)

            # 保存代码执行结果
            with open ('MBPPGen/data/shorten_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data/shorten_persona.jsonl', 'MBPPGen/data/'+ self.time +'/shorten_persona_'+self.time+'.jsonl')
        os.rename('MBPPGen/data/shorten_persona_result.jsonl', 'MBPPGen/data/'+ self.time +'/shorten_persona_result_'+self.time+'.jsonl')

if __name__ == "__main__":

    ee = experiment(0.1)
    # ee.run_few_shot_common_persona(0, 427, 3)
    # ee.run_few_shot_compare(0, 427, 3)
    ee.run_shorten_persona(0, 427)