import subprocess
from datasets import load_dataset
import json
import os
import argparse

def run_test_case(program, input_data):
    try:
        """运行单个测试用例"""
        result = subprocess.run(['python', program], capture_output=True, text=True, timeout=3, input=input_data)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return "timed out", e
    except Exception as e:
        return "timed out", e

def load_test_cases():
    """从文件中读取测试用例"""
    ds = load_dataset("codeparrot/apps", "interview",trust_remote_code=True)
    i_o = [json.loads(ds["test"][i]["input_output"]) for i in range(500)]
    test_cases = [i_o[i]["inputs"] for i in range(500)]
    output_data = [i_o[i]["outputs"] for i in range(500)]
    program = [json.loads(ds["test"][i]["solutions"])[0] for i in range(500)]
    return test_cases, output_data

def test_single_problem(program_data, test_case, output_data, program):
    with open(program, 'w') as f:
        f.write(program_data)
    print(f"Running question test...")
    for i, test in enumerate(test_case):
        input_data = test.strip()
        output, error = run_test_case(program, input_data)
        print(f"Input:\n{input_data}")
        print(f"Output:\n{output}")
        if output.strip() != output_data[i].strip():
            print(f"Unexpected Output:\n{output}")
            return f"input {i} failed, output {output}", False
        if error:
            print(f"Error:\n{error}")
            return f"input {i} failed, output {error}", False
    print(f"Question test passed.")
    return "All Passed", True

def test_single_problem_passrate(program_data, test_case, output_data, program):
    with open(program, 'w') as f:
        f.write(program_data)
    # print(f"Running question test...")
    passed = 0
    for i, test in enumerate(test_case):
        input_data = test.strip()
        output, error = run_test_case(program, input_data)
        # print(f"Input:\n{input_data}")
        # print(f"Output:\n{output}")
        if output.strip() != output_data[i].strip():
            # print(f"Unexpected Output:\n{output}")
            # print(f"input {i} failed, output {output}")
            continue
        elif error:
            # print(f"Error:\n{error}")
            # print(f"input {i} failed, output {error}")
            continue
        else:
            passed += 1
    if passed == len(test_case):
        # print(f"Question test all passed.")
        return 1.0, True
    else:
        # print(f"Question test failed.")
        return passed/len(test_case), False
    

def test_all_passrate(start, size, path, program):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name+'/results'):
        os.makedirs(dir_name+'/results')
    file_name = "passrate_" + os.path.basename(path)
    with open(path, 'r') as f:
        data = f.readlines()
        program_data = [json.loads(d)['solution'] for d in data]
        test_cases, output_data = load_test_cases()
        for i in range(start, size):
            if os.path.exists(dir_name+'/results/'+file_name):
                with open(dir_name+'/results/'+file_name, 'r') as f1:
                    lines = f1.readlines()
                    if len(lines) > i:
                        continue
            passrate, result = test_single_problem_passrate(program_data[i], test_cases[i], output_data[i], program)
            print(f"Question {i}: {passrate}")
            with open (dir_name+'/results/'+file_name, 'a') as f1:
                f1.write(json.dumps({"question": i,"passrate": passrate, "result": result})+'\n')


def test_all(start, size, path, program):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name+'/results'):
        os.makedirs(dir_name+'/results')
    file_name = os.path.basename(path)
    with open(path, 'r') as f:
        data = f.readlines()
        program_data = [json.loads(d)['solution'] for d in data]
        test_cases, output_data = load_test_cases()
        for i in range(start, size):
            output, result = test_single_problem(program_data[i], test_cases[i], output_data[i], program)
            # print(f"Question {i}: {result}")
            with open (dir_name+'/results/'+file_name, 'a') as f1:
                f1.write(json.dumps({"question": i,"output": output, "result": result})+'\n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--path", type=str, default='APPSGen/data_deepseek/202408202331/compare_result_202408202331.jsonl')
    parser.add_argument("--program", type=str, default='test.py')
    parser.add_argument("--passrate", default=1)
    path = parser.parse_args().path
    start = parser.parse_args().start
    size = parser.parse_args().size
    program = parser.parse_args().program
    passrate = parser.parse_args().passrate
    if passrate == 1:
        test_all_passrate(start, size, path, program)
    else:
        test_all(start, size, path, program)
