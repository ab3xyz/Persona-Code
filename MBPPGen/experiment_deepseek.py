import json
import os
import requests
import re
from datasets import load_dataset
from execute import CodeExecutor
from personality import personaGen
from datetime import datetime
import random

class deepseek_experiment:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("MBPPGen/data_deepseek/"+self.time):
            os.makedirs("MBPPGen/data_deepseek/"+self.time)
        return   
    
    def run_identity_to_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start+size):

            persona_log = {}
            realworld_prompt = personas.generate_realworld_problem_prompt(prompt[i])
            prompt_response = personas.send_deepseek_request(realworld_prompt, type="chat")
            persona_log["realworld"] = dict(prompt_response)["content"]
            identity_prompt = personas.generate_identity_prompt(persona_log["realworld"])
            prompt_response = personas.send_deepseek_request(identity_prompt, type="chat")
            persona_log["identity"] = dict(prompt_response)["content"]
            persona_prompt = personas.generate_persona_on_identities(persona_log["identity"])
            prompt_response = personas.send_deepseek_request(persona_prompt, type="chat")
            persona_log["persona"] = dict(prompt_response)["content"]

            # print("Persona:",personas.parse_persona(dict(prompt_response)["content"]))

            
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, persona_log['persona'], code[i])
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))


            with open ('MBPPGen/data_deepseek/identity_persona.jsonl', 'a') as f:
                json_str = json.dumps(dict(persona_log))
                f.write(json_str+'\n')
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)


            with open ('MBPPGen/data_deepseek/identity_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/identity_persona.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/identity_persona_'+self.time+'.jsonl')
        os.rename('MBPPGen/data_deepseek/identity_persona_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/identity_persona_result_'+self.time+'.jsonl')
    
    
    
    
    def run_persona(self,start,size):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):

            persona_prompt = personas.generate_personality_prompt(prompt[i])
            prompt_response = personas.send_deepseek_request(persona_prompt)
            # print("Persona:",personas.parse_persona(dict(prompt_response)["content"]))

            with open ('MBPPGen/data_deepseek/persona.jsonl', 'a') as f:
                json_str = json.dumps(dict(prompt_response))
                f.write(json_str+'\n')
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, dict(prompt_response)["content"], code[i])
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data_deepseek/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')


            with open ('MBPPGen/data_deepseek/persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/persona.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/persona_'+self.time+'.jsonl')
        os.rename('MBPPGen/data_deepseek/persona_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/persona_result_'+self.time+'.jsonl')

    def run_compare(self, start, size):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt=prompt[i], test=testcode, code=code[i])
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data_deepseek/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')


            with open ('MBPPGen/data_deepseek/compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/compare_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/compare_result_'+self.time+'.jsonl')
    
    def run_random_persona(self, start, size):
        personas = personaGen(self.temperature)
        lines = []
        with open ("MBPPGen/data_deepseek/202407190418/identity_persona_202407190418.jsonl", 'r') as f:
            lines = f.readlines()
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            # mbti = ""

            # persona = json.loads(lines[i])
            # mbti = persona["persona"]
            # Define the MBTI types as a list
            mbti_types = ["ISTJ", "ISTP", "ISFJ", "ISFP", "INFJ", "INFP", "INTJ", "INTP",
                        "ESTP", "ESTJ", "ESFJ", "ESFP", "ENFP", "ENFJ", "ENTP", "ENTJ"]
            # Generate a random index within the range of MBTI types
            random_index = random.randrange(len(mbti_types))
            # Select the MBTI type at the random index
            mbti = "ESTJ"
            # Print the randomly generated MBTI type
            print("mbti:", mbti)

            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, mbti, code[i])
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data_deepseek/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')


            with open ('MBPPGen/data_deepseek/ESTJ_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result), "mbti": mbti})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/ESTJ_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/ESTJ_result_'+self.time+'.jsonl')
    
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
            code_response = personas.send_deepseek_request(code_prompt)
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data_deepseek/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')


            with open ('MBPPGen/data_deepseek/common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/common_persona_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/common_persona_result_'+self.time+'.jsonl')


    def run_cot_compare(self,start, size):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_cot_prompt(prompt=prompt[i], test=testcode, code=code[i])
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data_deepseek/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')


            with open ('MBPPGen/data_deepseek/cot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/cot_compare_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/cot_compare_result_'+self.time+'.jsonl')

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
            code_response = personas.send_deepseek_request(code_prompt)
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)


            with open ('MBPPGen/data_deepseek/cot_common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/cot_common_persona_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/cot_common_persona_result_'+self.time+'.jsonl')

    def run_few_shot_compare(self, start, size, shot = 3):
        personas = personaGen(self.temperature)
        prompt, test, code = personas.get_original_data_with_code()
        for i in range(start, start + size):
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_few_shot_prompt(prompt=prompt[i], test=testcode, code=code[i], shot=shot)
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)
            # with open ('MBPPGen/data_deepseek/code.jsonl', 'a') as f:
            #     json_str = json.dumps(dict(code_response))
            #     f.write(json_str+'\n')


            file_name = '/few_shot_compare_result_'
            if shot == 3:
                file_name = '/three_shot_compare_result_'
            elif shot == 1:
                file_name = '/one_shot_compare_result_'
            with open ('MBPPGen/data_deepseek/few_shot_compare_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/few_shot_compare_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time + file_name +self.time+'.jsonl')

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
            code_response = personas.send_deepseek_request(code_prompt)
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)


            file_name = '/few_shot_common_persona_result_'
            if shot == 3:
                file_name = '/three_shot_common_persona_result_'
            elif shot == 1:
                file_name = '/one_shot_common_persona_result_'
            with open ('MBPPGen/data_deepseek/few_shot_common_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"response": dict(code_response)["content"], "code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/few_shot_common_persona_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time + file_name +self.time+'.jsonl')
    
    def run_shorten_persona(self, start, size):
        personas = personaGen(self.temperature)
        prompt, test = personas.get_original_data()
        with open ("MBPPGen/persona.jsonl", 'r') as f:
            lines = f.readlines()
            personality = [json.loads(line)["content"] for line in lines]
        for i in range(start, start+size):

            
            testcode = "\n".join(test[i])
            code_prompt = personas.generate_code_prompt(prompt[i], testcode, personality[i])
            code_response = personas.send_deepseek_request(code_prompt)
            # print("code:",personas.parse_code(dict(code_response)["content"]))
            

            exci = CodeExecutor()
            generated_code = personas.parse_code(dict(code_response)["content"]) + "\n" + testcode
            print("generated code:%d ",i, generated_code)
            exe_result = exci.execute_code(generated_code)
            print("execute result", exe_result)


            with open ('MBPPGen/data_deepseek/shorten_persona_result.jsonl', 'a') as f:
                json_str = json.dumps({"code": generated_code, "result":str(exe_result), "success":exci.check_result(exe_result)})
                f.write(json_str+'\n')
        os.rename('MBPPGen/data_deepseek/shorten_persona.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/shorten_persona_'+self.time+'.jsonl')
        os.rename('MBPPGen/data_deepseek/shorten_persona_result.jsonl', 'MBPPGen/data_deepseek/'+ self.time +'/shorten_persona_result_'+self.time+'.jsonl')

if __name__ == "__main__":
    ee = deepseek_experiment(0.1)
    ee.run_shorten_persona(248, 427 - 248)