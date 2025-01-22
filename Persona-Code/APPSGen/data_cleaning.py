import subprocess
from datasets import load_dataset
import json
import os
import argparse
from personality import personaGen
import re


class data_cleaning:
    def __init__(self) -> None:
        pass

    def parse_code(self, code):
        if "```" not in code:
            return code
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
                return ""
    
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
                            mid = d[-1]["content"]
                            mid = self.parse_code(mid)
                            f.write(json.dumps({"solution": mid})+'\n')
                            

if __name__ == "__main__":
    dc = data_cleaning()
    dc.cleaning("APPSGen/data_llama3_1/202409020737")