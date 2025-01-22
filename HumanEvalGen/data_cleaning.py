import subprocess
from datasets import load_dataset
import json
import os
import argparse
import re


class data_cleaning:
    def __init__(self) -> None:
        pass

    def parse_code(self, code):
        pattern = r'```python(.*?)```'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            new_pattern = r'```(.*?)```'
            new_match = re.search(new_pattern, code, re.DOTALL)
            if new_match:
                return new_match.group(1).strip()
            else:
                new_new_pattern = r"\[PYTHON\](.*?)\[/PYTHON\]"
                new_new_match = re.search(new_new_pattern, code, re.DOTALL)
                if new_new_match:
                    return new_new_match.group(1).strip()
                else:
                    return code.strip()
    
    def cleaning(self,base_path):
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.jsonl'):
                    original_data = []
                    with open(os.path.join(root, file), 'r') as f:
                        data = f.readlines()
                        original_data = [json.loads(d) for d in data]
                    with open(os.path.join(root, file), 'w') as f:
                        for d in original_data:
                            task_id = d["task_id"]
                            mid = d["completion"][-1]["content"]
                            mid = self.parse_code(mid)
                            f.write(json.dumps({"task_id": task_id, "solution": mid})+'\n')
                            

if __name__ == "__main__":
    dc = data_cleaning()
    # dc.cleaning("HumanEvalGen/data_llama3_1")
    # dc.cleaning("HumanEvalGen/data_7B")
    # dc.cleaning("HumanEvalGen/data_13B")