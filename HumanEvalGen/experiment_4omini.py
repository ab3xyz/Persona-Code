import os
from datetime import datetime
from humanEvalGen import humanEvalGen
import json
from human_eval.data import write_jsonl, read_problems
class experiment_4omini:
    def __init__(self, temperature) -> None:
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime('%Y%m%d%H%M')
        self.time = formatted_datetime
        self.temperature = temperature
        if not os.path.exists("HumanEvalGen/data_4omini/"+self.time):
            os.makedirs("HumanEvalGen/data_4omini/"+self.time)
        return
    
    def identity_to_persona_method(self, prompt):
        # 产生人格的prompt
        persona_log = {}
        heg = humanEvalGen(self.temperature)
        realworld_prompt = heg.generate_realworld_problem_prompt(prompt)
        prompt_response = heg.send_4omini_request(realworld_prompt)
        persona_log["realworld"] = dict(prompt_response)["content"]
        identity_prompt = heg.generate_identity_prompt(persona_log["realworld"])
        prompt_response = heg.send_4omini_request(identity_prompt)
        persona_log["identity"] = dict(prompt_response)["content"]
        persona_prompt = heg.generate_persona_on_identities(persona_log["identity"])
        prompt_response = heg.send_4omini_request(persona_prompt)
        persona_log["persona"] = dict(prompt_response)["content"]

        # 产生代码的prompt
        code_prompt = heg.generate_code_prompt(prompt, persona_log["persona"])
        code_response = heg.send_4omini_request(code_prompt)

        with open ("HumanEvalGen/data_4omini/"+self.time+"/identity_persona.jsonl", 'a') as f:
            json_str = json.dumps(dict(persona_log))
            f.write(json_str+'\n')
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])

    def compare_method(self, prompt):
        heg = humanEvalGen(self.temperature)
        code_prompt = heg.generate_code_prompt(prompt)
        code_response = heg.send_4omini_request(code_prompt)
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])
    
    def random_persona_method(self, prompt):
        heg = humanEvalGen(self.temperature)
        code_prompt = heg.generate_code_prompt(prompt, "ESTJ")
        code_response = heg.send_4omini_request(code_prompt)
        print(heg.parse_code(dict(code_response)["content"]))
        return heg.parse_code(dict(code_response)["content"])
    

    def persona_method(self, prompt):
        heg = humanEvalGen(self.temperature)
        persona_log = {}
        persona_prompt = heg.generate_personality_prompt(prompt)
        prompt_response = heg.send_4omini_request(persona_prompt)
        persona_log["persona"] = dict(prompt_response)["content"]
        code_prompt = heg.generate_code_prompt(prompt, persona_log["persona"])
        code_response = heg.send_4omini_request(code_prompt)

        with open ("HumanEvalGen/data_4omini/"+self.time+"/persona.jsonl", 'a') as f:
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
            code_response = heg.send_4omini_request(code_prompt)
            print(heg.parse_code(dict(code_response)["content"]))
            return heg.parse_code(dict(code_response)["content"])
    

    def run_identity_to_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_4omini/"+self.time+"/identity_persona_result.jsonl"
        heg.experiment(self.identity_to_persona_method, path, start)

    def run_compare(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_4omini/"+self.time+"/compare.jsonl"
        heg.experiment(self.compare_method, path, start)

    def run_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_4omini/"+self.time+"/persona_result.jsonl"
        heg.experiment(self.persona_method,path, start)

    def run_random_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_4omini/"+self.time+"/random_persona_result.jsonl"
        heg.experiment(self.random_persona_method, path, start)

    def run_common_persona(self, start = 0, end = 164):
        heg = humanEvalGen(self.temperature)
        path = "HumanEvalGen/data_4omini/"+self.time+"/common_persona_result.jsonl"
        heg.new_experiment(self.common_persona_method, path, start)

    def mending(self, result_path, persona_path):
        with open(result_path, 'r') as f:
            results = f.readlines()
        with open(persona_path, 'r') as f:
            personas = f.readlines()
        assert len(results) == len(personas), "Not equal length"
        problems = read_problems()
        heg = humanEvalGen(0)
        for i in range(len(results)):
            result = json.loads(results[i])
            persona = json.loads(personas[i])
            if result["completion"] == "":
                prompt = problems[result["task_id"]]["prompt"]
                tests = problems[result["task_id"]]["test"]
                entry_point = problems[result["task_id"]]["entry_point"]
                code_prompt = heg.generate_code_prompt(prompt, persona["persona"])
                code_response = heg.send_4omini_request(code_prompt)
                with open("HumanEvalGen/data_4omini/mending/mending"+str(i)+".py", 'w') as f:
                    f.write(prompt)
                    f.write(dict(code_response)["content"])
                    f.write(tests)
                    f.write(f"check({entry_point})")

        return


if __name__ == "__main__":
    exp = experiment_4omini(0)
    # exp.run_compare()
    # exp.run_identity_to_persona()
    exp.run_common_persona()
    # exp.run_random_persona()
    # exp.mending("HumanEvalGen/data_4omini/202407231637/persona_result.jsonl", "HumanEvalGen/data_4omini/202407231637/persona.jsonl")
