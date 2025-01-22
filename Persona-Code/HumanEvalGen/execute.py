from human_eval.evaluation import evaluate_functional_correctness
import json

import subprocess
class CodeExecutor:
    def __init__(self):
        return

    def execute_code(self, code_str):

        # 执行temp.py文件并捕获输出
        try :
            with open('temp.py', 'w') as file:
                file.write(code_str)
            result = subprocess.run(['python', 'temp.py'], capture_output=True, text=True, timeout=5)
            return result
        except subprocess.TimeoutExpired as e:
            return e
        except Exception as e:
            return "timed out"
    

    
    def check_result(self, result):
        """
        Checks the result of code execution for errors.
        
        Parameters:
        result (dict): The dictionary containing the local variables after code execution.
        
        Returns:
        str: A string indicating the status of the code execution.
        """
        if "timed out" in str(result):
            return 0
        elif len(str(result.stderr)) > 0:
            return 0
        return 1 # Successful execution