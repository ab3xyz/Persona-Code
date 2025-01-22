from human_eval.data import write_jsonl, read_problems
import json
from openai import OpenAI
import re
from execute import CodeExecutor
import os

class mendingHumanEval:
    def __init__(self) -> None:
        pass

    def remove_starting_lines(self, code_str):
        lines = code_str.split('\n')
        cleaned_lines = []
        in_function = False

        for line in lines:
            if in_function:
                cleaned_lines.append(line)
            if line.strip().startswith('def '):
                in_function = True
        if not cleaned_lines:
            return code_str
        return '\n'.join(cleaned_lines)

    def mending(self, path):
        problems = read_problems()
        results = []
        exc = CodeExecutor()
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                results.append(json.loads(line))
        assert len(results) == len(problems), "Not equal length"
        for i in range(len(results)):
            if results[i]["success"] == 1:
                continue
            prompt = problems[results[i]["task_id"]]["prompt"]
            code = results[i]["completion"]
            code = prompt + self.remove_starting_lines(code)
            # 重新执行代码获得结果
            task_id = results[i]["task_id"]
            check_program = code + "\n" + problems[results[i]["task_id"]]["test"] + "\n" + f"check({problems[task_id]['entry_point']})"
            exc_ret = exc.execute_code(check_program)
            exc_result = exc.check_result(exc_ret)
            if exc_result == 1:
                results[i]["completion"] = code
                results[i]["success"] = 1
                results[i]["result"] = str(exc_ret)
        dir_path = os.path.dirname(path)
        file_name = os.path.basename(path)

        new_path = dir_path + "/mending/"
        new_file_name = "mending_" + file_name
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        new_path += new_file_name
        write_jsonl(new_path, results)

if __name__ == "__main__":
    mend = mendingHumanEval()
    # mend.mending("HumanEvalGen/data_qwen/202407231520/compare.jsonl")
    # mend.mending("HumanEvalGen/data_deepseek/202408251536/compare.jsonl")
    mend.mending("HumanEvalGen/data_deepseek/202408251536/common_persona_result.jsonl")
    mend.mending("HumanEvalGen/data_deepseek/202408251536/persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_qwen/202408231253/common_persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_codestral/202408231135/common_persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_codestral/202408230319/compare.jsonl")
    # mend.mending("HumanEvalGen/data_codestral/202408230319/persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_4omini/202407261724/compare.jsonl")
    # mend.mending("HumanEvalGen/data_4omini/202407262157/persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_4omini/202408230320/common_persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_4o/202408230117/compare.jsonl")
    # mend.mending("HumanEvalGen/data_4o/202408230117/persona_result.jsonl")
    # mend.mending("HumanEvalGen/data_qwen/202408252022/compare.jsonl")
    # mend.mending("HumanEvalGen/data_qwen/202408252022/persona_result.jsonl")