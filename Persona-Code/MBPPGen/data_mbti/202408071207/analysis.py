import os
import json
def walk_files(root_dir):
    str_a = "jsonl"

    pairs = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if str_a in file:
                pairs.append(os.path.join(root, file))
    return pairs

def load_code_from_file(path):
    code_strs = []
    results = []
    cnt = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            mid = json.loads(line)
            code_strs.append(mid["code"])
            results.append(mid["success"])
    return code_strs, results

def MBTI_analysis(files):
    root_dir = os.path.dirname(files[0])
    counts = {}
    for file in files:
        code_strs, results = load_code_from_file(file)
        cnt = 0
        for result in results:
            if result == 1:
                cnt += 1
        print("The success rate of {} is {}".format(file, cnt/len(results)))
        counts[file] = cnt/len(results)
        with open(root_dir + "\\analysis.txt", "a") as f:
            file_name = os.path.basename(file)
            f.write(json.dumps({"MBTI":file_name, "success_rate":cnt/len(results)}) + '\n')

    max = 0
    for count in counts:
        if counts[count] > max:
            max = counts[count]
            max_file = count
            print("The max success rate is {} in {}".format(max, max_file))
    min = 1
    for count in counts:
        if counts[count] < min:
            min = counts[count]
            min_file = count
            print("The min success rate is {} in {}".format(min, min_file))

if __name__ == '__main__':
    files = walk_files("MBPPGen\\data_qwen\\202408071207")
    MBTI_analysis(files)
