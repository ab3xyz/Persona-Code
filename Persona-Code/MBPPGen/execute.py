import subprocess
class CodeExecutor:
    def __init__(self):
        return
    
    # def execute_code(self, code_str):
    #     """
    #     Executes a given string of Python code.
        
    #     Parameters:
    #     code_str (str): The string containing the Python code to be executed.
        
    #     Returns:
    #     dict: A dictionary containing the local variables after code execution.
    #     """
    #     local_vars = {}
    #     try:
    #         # Execute the code in a specific local context
    #         with open("temp", "w") as f:
    #             f.write(code_str)
    #         exec(code_str, globals(), local_vars)
    #     except Exception as e:
    #         return {"error": str(e)}
        
    #     return local_vars

    def execute_code(self, code_str):

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
        if len(str(result.stderr)) > 0:
            return 0
        return 1 # Successful execution
