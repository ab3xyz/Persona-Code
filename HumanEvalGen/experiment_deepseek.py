import os
from datetime import datetime
from humanEvalGen import humanEvalGen
import json

class experiment_deepseek:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("HumanEvalGen/data_deepseek/"+self.time):
            os.makedirs("HumanEvalGen/data_deepseek/"+self.time)
        return
    
    def identity_to_persona_method(self, prompt):
        # 产生人格的prompt
        persona_log = {}
        heg = humanEvalGen(self.temperature)
        realworld_prompt = heg.generate_realworld_problem_prompt(prompt)
        prompt_response = heg.send_deepseek_request(realworld_prompt, type="chat")
        persona_log["realworld"] = dict(prompt_response)["content"]
        identity_prompt = heg.generate_identity_prompt(persona_log["realworld"])
        prompt_response = heg.send_deepseek_request(identity_prompt, type="chat")
        persona_log["identity"] = dict(prompt_response)["content"]
        persona_prompt = heg.generate_persona_on_identities(persona_log["identity"])
        prompt_response = heg.send_deepseek_request(persona_prompt, type="chat")
        persona_log["persona"] = dict(prompt_response)["content"]

        # 产生代码的prompt
        code_prompt = heg.generate_code_prompt(prompt, persona_log["persona"])
        code_response = heg.send_deepseek_request(code_prompt)

        with open ("HumanEvalGen/data_deepseek/"+self.time+"/identity_persona.jsonl", 'a') as f:
            json_str = json.dumps(dict(persona_log))
            f.write(json_str+'\n')
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])

    def compare_method(self, prompt):
        heg = humanEvalGen(self.temperature)
        code_prompt = heg.generate_code_prompt(prompt)
        code_response = heg.send_deepseek_request(code_prompt)
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])
    
    def random_persona_method(self, prompt):
        heg = humanEvalGen(self.temperature)
        code_prompt = heg.generate_code_prompt(prompt, "ESTJ")
        code_response = heg.send_deepseek_request(code_prompt)
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])
    

    def persona_method(self, prompt):
        heg = humanEvalGen(self.temperature)
        persona_log = {}
        persona_prompt = heg.generate_personality_prompt(prompt)
        prompt_response = heg.send_deepseek_request(persona_prompt, type="chat")
        persona_log["persona"] = dict(prompt_response)["content"]
        code_prompt = heg.generate_code_prompt(prompt, persona_log["persona"])
        code_response = heg.send_deepseek_request(code_prompt)

        with open ("HumanEvalGen/data_deepseek/"+self.time+"/persona.jsonl", 'a') as f:
            json_str = json.dumps(dict(persona_log))
            f.write(json_str+'\n')
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])
    

    def common_persona_method(self, prompt, position):
        heg = humanEvalGen(self.temperature)
        with open("HumanEvalGen/persona.jsonl", 'r') as f:
            data = f.readlines()
            personas = [json.loads(persona)["persona"] for persona in data]
            persona = personas[position]
            code_prompt = heg.generate_code_prompt(prompt, persona)
            code_response = heg.send_codestral_request(code_prompt)
            print(heg.parse_code(dict(code_response)["content"]))
            return heg.parse_code(dict(code_response)["content"])

    def run_identity_to_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_deepseek/"+self.time+"/identity_persona_result.jsonl"
        heg.experiment(self.identity_to_persona_method, path)

    def run_compare(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_deepseek/"+self.time+"/compare.jsonl"
        heg.experiment(self.compare_method, path)

    def run_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_deepseek/"+self.time+"/persona_result.jsonl"
        heg.experiment(self.persona_method,path, start)

    def run_random_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_deepseek/"+self.time+"/random_persona_result.jsonl"
        heg.experiment(self.random_persona_method, path)

    def run_common_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_deepseek/"+self.time+"/common_persona_result.jsonl"
        heg.new_experiment(self.common_persona_method, path, start)

if __name__ == "__main__":
    exp = experiment_deepseek(0)
    # exp.run_identity_to_persona()
    exp.run_common_persona()
    exp.run_compare()
    # exp.run_persona()
    # exp.run_random_persona()
