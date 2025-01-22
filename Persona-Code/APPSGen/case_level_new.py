import subprocess
from datasets import load_dataset
import json
import os
import argparse
import re
import matplotlib.pyplot as plt


def load_test_cases():
    """从文件中读取测试用例"""
    ds = load_dataset("codeparrot/apps", "interview",trust_remote_code=True)
    i_o = [json.loads(ds["test"][i]["input_output"]) for i in range(500)]
    test_cases = [i_o[i]["inputs"] for i in range(500)]
    output_data = [i_o[i]["outputs"] for i in range(500)]
    program = [json.loads(ds["test"][i]["solutions"])[0] for i in range(500)]
    return test_cases, output_data


def compare_caselevel_passrate(path):
    compare_data = []
    common_persona_data = []
    test_cases, output_data = load_test_cases()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jsonl'):
                with open(os.path.join(root, file), 'r') as f:
                    print(os.path.join(root, file))
                    if "compare" in file:
                        data_mid = f.readlines()
                        compare_data = [json.loads(p) for p in data_mid]
                        # print(data_mid)
                    else:
                        data_mid = f.readlines()
                        common_persona_data = [json.loads(p) for p in data_mid]
                        # print(len(compare_data))
    assert len(compare_data) == len(common_persona_data)
    case_level_passrate = []
    for i in range(len(compare_data)):
        compare_pass = float(compare_data[i]["passrate"])
        common_persona_pass = float(common_persona_data[i]["passrate"])
        result_mid = {"compare": compare_pass, "common_persona": common_persona_pass, "improvement": common_persona_pass - compare_pass}
        case_level_passrate.append(result_mid)
        
    with open(os.path.join(path + "/case_level_passrate.jsonl"), 'w') as f:
        for item in case_level_passrate:
            f.write(json.dumps(item) + '\n')
    print(sum([item['improvement'] for item in case_level_passrate]) / len(case_level_passrate))
    return case_level_passrate
    # draw_figure(case_level_passrate, path)
    
def call_all_caselevel(path):
    try:
        dirs = os.listdir(path)
        result = []
        for dir in dirs:
            if os.path.isdir(os.path.join(path, dir)):
                result_mid = compare_caselevel_passrate(os.path.join(path, dir))
                improved_cases = len([item for item in result_mid if item['improvement'] > 0])
                worsened_cases = len([item for item in result_mid if item['improvement'] < 0])
                avg_improvement = sum([item['improvement'] for item in result_mid]) / len(result_mid)
                improved_rate = sum([item['improvement'] for item in result_mid if item['improvement'] > 0]) / improved_cases
                worsened_rate = sum([item['improvement'] for item in result_mid if item['improvement'] < 0]) / worsened_cases
                incorrect_to_correct = len([item for item in result_mid if item['common_persona'] == 1.0 and item["compare"] != 1.0])
                correct_to_incorrect = len([item for item in result_mid if item['common_persona'] != 1.0 and item["compare"] == 1.0])
                result.append({"dir": dir, "improved_cases": improved_cases, "worsened_cases": worsened_cases, "avg_improvement": avg_improvement, "improved_rate": improved_rate, "worsened_rate": worsened_rate, "incorrect_to_correct": incorrect_to_correct, "correct_to_incorrect": correct_to_incorrect})
        with open(os.path.join(path + "/all_improvement.jsonl"), 'w') as f:
            for item in result:
                f.write(json.dumps(item) + '\n')
    except Exception as e:
        print(e)
        return
    
def compare_two_caselevel(path1, path2):
    data1 = []
    data2 = []
    with open(path1, 'r') as f:
        data1 = f.readlines()
    with open(path2, 'r') as f:
        data2 = f.readlines()
    case_level_passrate1 = [json.loads(p) for p in data1]
    case_level_passrate2 = [json.loads(p) for p in data2]
    assert len(case_level_passrate1) == len(case_level_passrate2)
    case_level_passrate = []
    cnt = 0
    for i in range(len(case_level_passrate1)):
        if case_level_passrate1[i]["result"] == case_level_passrate2[i]["result"]:
            continue
        cnt += 1
    return cnt


def draw_figure(case_level_passrate, path):
    # 假设你的数据存储在一个 JSON 文件中，文件名为 data.json
    data = case_level_passrate
    # 提取所有的 improvement 值
    improvements = [item['improvement'] for item in data]

    # 设置图形大小
    plt.figure(figsize=(10, 5))

    # 绘制柱状图
    plt.bar(range(len(improvements)), improvements, width=0.8, align='center')

    # 设置 x 轴标签
    plt.xticks(range(len(improvements)), [f'Item {i+1}' for i in range(len(improvements))], rotation=90)

    # 设置标题和标签
    plt.title('Improvement Values')
    plt.xlabel('Items')
    plt.ylabel('Improvement')

    # 显示图形
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(path + "/case_level_improvement.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='APPSGen/caselevel_study/4o')
    path = parser.parse_args().path
    # compare_caselevel_passrate(path)
    call_all_caselevel(path)
    # print(compare_two_caselevel("APPSGen/data_llama3_1/202408300008/results/common_persona_result_202408300008.jsonl", "APPSGen/data_llama3_1/202409020737/results/common_persona_result_202409020737.jsonl"))
    # print(compare_two_caselevel("APPSGen/data_llama3_1/202408300652/results/compare_result_202408300652.jsonl", "APPSGen/data_llama3_1/202408301335/results/compare_result_202408301335.jsonl"))
    # print(compare_two_caselevel("APPSGen/data_qwen/202408300046/results/compare_result_202408300046.jsonl", "APPSGen/data_qwen/202408301055/results/compare_result_202408301055.jsonl"))