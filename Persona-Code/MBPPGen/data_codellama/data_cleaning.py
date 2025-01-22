import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from execute import CodeExecutor
from personality import personaGen
import re

class data_cleaning:
    def __init__(self) -> None:
        self.persona = personaGen(0)
        pass


    def parse_code(self, code):
        if "```" not in code:
            return code[2:]
        pattern = r'```(.*?)```'
        py_pattern = r'```python(.*?)```'
        match = re.search(pattern, code, re.DOTALL)
        py_match = re.search(py_pattern, code, re.DOTALL)
        if py_match:
            return py_match.group(1).strip()
        else:
            if match:
                return match.group(1).strip()
            return ""

    def load_data(self, path):
        code_strs = []
        with open(path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                data = json.loads(line)
                # 获取最后一部分中的代码
                code_content = data["code"][-1]["content"]
                code_strs.append(code_content)
        return code_strs
    
    def clean_data(self, path):
        code_strs = self.load_data(path)
        code_executor = CodeExecutor()
        persona = self.persona
        prompt, test = persona.get_original_data()
        for i, code_str in enumerate(code_strs):
            code = self.parse_code(code_str)
            code += "\n" + "\n".join(test[i])
            result = code_executor.execute_code(code)
            print(result)
            success = code_executor.check_result(result)
            dir_path = os.path.dirname(path)
            file_name = os.path.basename(path)
            with open(dir_path + "/result_" + file_name, 'a') as f:
                f.write(json.dumps({"code": code, "result": str(result) ,"success": success}) + "\n")
            
    def walk_path(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                print(os.path.join(root, file))
                if file.endswith(".jsonl") and not file.startswith("result_"):
                    if os.path.exists(os.path.join(root, "result_" + file)):
                        continue
                    self.clean_data(os.path.join(root, file))

if __name__ == "__main__":
    data_cleaning = data_cleaning()
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/compare_result_7B_2.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/persona_result_7B_2.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/compare_result_7B_3.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/persona_result_7B_3.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/compare_result_13B_2.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/persona_result_13B_2.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/compare_result_13B_3.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/persona_result_13B_3.jsonl")
    
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/compare_result_7B_5.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/persona_result_7B_5.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/compare_result_13B_5.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/persona_result_13B_5.jsonl")    
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/compare_result_7B_6.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/7B/multiple/persona_result_7B_6.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/compare_result_13B_6.jsonl")
    # data_cleaning.clean_data("MBPPGen/data_codellama/13B/multiple/persona_result_13B_6.jsonl")
    data_cleaning.walk_path("MBPPGen/data_codellama/reverse_new")