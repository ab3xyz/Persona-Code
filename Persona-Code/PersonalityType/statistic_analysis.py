import json
import math
from math import sqrt
from scipy.stats import norm

def load_data(path):
    data = []
    with open(path, 'r') as f:
        mid = f.readlines()
        for line in mid:
            data.append(json.loads(line))
    return data

def check_proportion(data):
    task_cnt = {"MBPP Sanitized": 427, "MBPP+": 399, "HumanEval+": 164, "APPS": 500}
    for i in range(28):
        dataset = data[2*i]["dataset"]
        model = data[2*i]["model"]
        task_count = task_cnt[dataset]
        avg_p = (data[2*i]["score"] + data[2*i + 1]["score"]) / 2
        test = (data[2*i + 1]["score"] - data[2*i]["score"]) / sqrt((avg_p) * (1-avg_p) * (2/(task_count * 10)))
        print(test)
        p_value_two_sided = 2 * (1 - norm.cdf(abs(test)))

        with open("PersonalityType/proportion_test.jsonl", "a") as f:
            f.write(json.dumps({"dataset": dataset, "model": model, "test": test, "p_value": p_value_two_sided}) + "\n")

def main():
    data = load_data("PersonalityType/statistic.jsonl")
    check_proportion(data)
            

if __name__ == "__main__":
    main()