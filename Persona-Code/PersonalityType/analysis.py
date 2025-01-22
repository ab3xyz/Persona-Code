import json
import os
from openai import OpenAI
import requests
import re
from datasets import load_dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

class analysis:

    def __init__(self):
        pass
    
    def send_mini4o_request(self, messages):
        client = OpenAI(api_key=self.OPENAI_API_KEY)
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        )

        return completion.choices[0].message
    
    def gain_personality(self, path):
        file = os.path.basename(path)
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                personality = data["content"]
                prompts = []
                prompts.append({
                    "role": "system",
                    "content": "Please tell me the MBTI type generated in the following content. Just give me the four letters, no other information."
                })
                prompts.append({
                    "role": "user",
                    "content": personality
                })
                response = self.send_mini4o_request(prompts)
                print(dict(response)["content"])
                ans_file = "type_" + file
                with open(os.path.join(os.path.dirname(path), ans_file), 'a') as f:
                    f.write(dict(response)["content"] + "\n")

    def walk_dir_and_gain_personality(self, dir):
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".jsonl"):
                    if "type_" in file:
                        continue
                    if "type_" + file in files:
                        continue
                    path = os.path.join(root, file)
                    self.gain_personality(path)

    def draw_pie_figures(self, path):
        # 文件夹路径
        folder_path = path

        # 筛选以Type开头的文件
        type_files = [f for f in os.listdir(folder_path) if f.startswith('type') and f.endswith('.jsonl')]

        # 读取每个文件的内容
        for file_name in type_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # 统计每行内容的出现次数
                content_counter = Counter(lines)

                # 提取标签和数据
                labels = list(content_counter.keys())
                sizes = list(content_counter.values())
                # Seaborn 配色方案
                # colors = sns.color_palette("Set2")  # 使用 Seaborn 中的 muted 配色
                # colors = sns.color_palette("Spectral", n_colors=4)
                pastel = sns.color_palette("pastel") # 我觉得不错，浅蓝色很高级
                # colors = pastel
                # colors = ["#fc9272", "#7fcdbb", "#2c7fb8"]
                # colors = ["#2c7fb8", "#fc9272",  pastel[2]]  # 8 种颜色
                # colors = [pastel[0],pastel[2], "#fc9272"] # v1.0
                colors = ["#67a8cd",pastel[2], "#fc9272"] # v 1.1
                colors = ["#3581B7",pastel[2], "#fc9272"] # v 1.2
                colors = ["#7ED3F6",pastel[2], "#fc9272"] # v 1.3
                # colors = sns.color_palette("bright")
                # colors = sns.color_palette("dark")  # 使用 Seaborn 中的 dark 配色
                # colors = sns.color_palette("colorblind")

                # 创建饼状图
                plt.figure(figsize=(6, 6))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, 
                        wedgeprops={'edgecolor': 'None', 'width': 0.45}, textprops={'fontsize': 14})

                # 保证饼状图为圆形
                plt.axis('equal')
                
                # 保存图片
                output_image_path = os.path.join(folder_path, file_name.replace('.jsonl', '.png'))
                plt.savefig(output_image_path)

                print(f"饼状图已保存为: {output_image_path}")

    def draw_accumulate_column_figures(self, path):
        # # 文件夹路径
        # folder_path = path

        # # 筛选以Type开头的文件
        # type_files = [f for f in os.listdir(folder_path) if f.startswith('type') and f.endswith('.jsonl')]

        # # 读取每个文件的内容
        # for file_name in type_files:
        #     file_path = os.path.join(folder_path, file_name)
        #     with open(file_path, 'r') as file:
        #         lines = file.readlines()

        #         # 统计每行内容的出现次数
        #         content_counter = Counter(lines)

        #         # 提取标签和数据
        #         labels = list(content_counter.keys())
        #         sizes = list(content_counter.values())
        #         # Seaborn 配色方案
        #         # colors = sns.color_palette("Set2")  # 使用 Seaborn 中的 muted 配色
        #         colors = sns.color_palette("Spectral", n_colors=4)
        #         pastel = sns.color_palette("pastel")
        # 文件夹路径
        folder_path = path

        # 筛选以Type开头的文件
        type_files = [f for f in os.listdir(folder_path) if f.startswith('type') and f.endswith('.jsonl')]


        # 用于存储每个文件的内容统计
        file_contents = {}

        # 读取每个文件的内容并统计
        for file_name in type_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                content_counter = Counter(lines)
                file_contents[file_name] = content_counter

        # 获取所有不同的内容标签
        all_labels = set()
        for content in file_contents.values():
            all_labels.update(content.keys())

        # 将标签排序
        all_labels = sorted(all_labels)

        # 创建堆叠柱状图的数据
        num_files = len(type_files)
        num_labels = len(all_labels)
        data = [[file_contents[file_name].get(label, 0) for label in all_labels] for file_name in type_files]
        print(data)
        # 绘制堆叠柱状图
        # 设置柱子宽度和位置
        bar_width = 0.5
        categories = [file_name.replace('_persona.jsonl', '').replace('type_', "") for file_name in type_files]
        indices = np.arange(len(categories))
        type1 = [d[0] / sum(d) for d in data]
        type2 = [d[1] / sum(d) for d in data]
        type3 = [d[2] / sum(d) for d in data]
        # 绘制堆叠柱状图
        plt.bar(indices, type1, bar_width, label='Type 1', color='b')
        plt.bar(indices, type2, bar_width, bottom=type1, label='Type 2', color='g')
        plt.bar(indices, type3, bar_width, bottom=np.array(type1) + np.array(type2), label='Type 3', color='r')

        # 添加标签和标题
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.title('Stacked Bar Chart')
        plt.legend(loc='upper right')
        plt.xticks(indices, categories)


        # # 设置图例和标签
        # ax.set_xticks(indices)
        # ax.set_xticklabels(type_files, rotation=45, ha='right')
        # ax.set_ylabel('Count')
        # ax.set_title('Stacked Bar Chart of File Contents')
        # ax.legend()

        # 保存图片
        output_image_path = os.path.join(folder_path, 'stacked_bar_chart.png')
        plt.savefig(output_image_path, bbox_inches='tight')

        print(f"堆叠柱状图已保存为: {output_image_path}")

if __name__ == "__main__":
    analysis = analysis()
    # analysis.walk_dir_and_gain_personality("SE-personas/PersonalityType/personas")
    # analysis.draw_pie_figures("SE-personas/PersonalityType/personas")
    analysis.draw_accumulate_column_figures("SE-personas/PersonalityType/personas")