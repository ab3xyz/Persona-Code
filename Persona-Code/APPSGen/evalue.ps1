# # 定义你要并行执行的命令
# ..\.venv\Scripts\activate
# $commands = @(
#     "python APPSGen/evaluate.py --path APPSGen/data_qwen/202408250016/common_persona_result_202408250016.jsonl --program test1.py",
#     "python APPSGen/evaluate.py --path APPSGen/data_deepseek/202408250254/common_persona_result_202408250254.jsonl --program test2.py",
#     "python APPSGen/evaluate.py APPSGen/data_codestral/202408250137/common_persona_result_202408250137.jsonl --program test3.py",
#     "python APPSGen/evaluate.py APPSGen/data_4omini/202408242258/common_persona_result_202408242258.jsonl --program test4.py"
# )

# # 并行执行所有命令
# $jobs = @()
# foreach ($command in $commands) {
#     $jobs += Start-Job -ScriptBlock {
#         param($cmd)
#         Invoke-Expression $cmd
#     } -ArgumentList $command
# }

# # 等待所有任务完成
# $jobs | ForEach-Object { 
#     $_ | Wait-Job | Receive-Job
# }

# # 清理完成的任务
# $jobs | ForEach-Object { Remove-Job $_ }
# 获取当前的工作目录
$CurrentDir = Get-Location

# 设置虚拟环境的相对路径
$venvPath = "$CurrentDir/../.venv/Scripts/activate"

# 定义要并行执行的命令
$commands = @(
#     "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_4o/202408241715/compare_result_202408241715.jsonl --program test1.py",
#     "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_4o/202408241715/persona_result_202408241715.jsonl --program test2.py",
#     "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_4omini/202408220009/compare_result_202408220009.jsonl --program test3.py",
#     "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_4omini/202408242258/common_persona_result_202408242258.jsonl --program test4.py",

    "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_13B/202408310007/common_persona_result_202408310007.jsonl --program test5.py"
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_13B/202408310007/compare_result_202408310007.jsonl --program test6.py",
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_codestral/202408220011/compare_result_202408220011.jsonl --program test7.py"
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_codestral/202408250137/common_persona_result_202408250137.jsonl --program test8.py",

    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_deepseek/202408202331/compare_result_202408202331.jsonl --program test9.py",
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_deepseek/202408250254/common_persona_result_202408250254.jsonl --program test10.py"
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_llama3_1/202408300008/common_persona_result_202408300008.jsonl --program test11.py",
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_llama3_1/202408300008/compare_result_202408300008.jsonl --program test12.py",

    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_qwen/202408250016/common_persona_result_202408250016.jsonl --program test13.py",
    # "python $CurrentDir/APPSGen/evaluate.py --path $CurrentDir/APPSGen/data_qwen/202408211347/compare_result_202408211347.jsonl --program test14.py"
)

# 激活虚拟环境
. $venvPath

# 并行执行所有命令
$jobs = @()
foreach ($command in $commands) {
    $jobs += Start-Job -ScriptBlock {
        param($cmd)
        Invoke-Expression $cmd
    } -ArgumentList $command
}

# 等待所有任务完成
$jobs | ForEach-Object { 
    $_ | Wait-Job | Receive-Job
}

# 清理完成的任务
$jobs | ForEach-Object { Remove-Job $_ }
