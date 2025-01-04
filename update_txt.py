# 这个代码的作用是对于分段的牙齿数据进行tran.txt和val.txt的追加
import os
# 定义源目录和目标目录
source_dir = ''
target_dir = ''
# 定义要处理的文件名
files_to_process = ['val.txt', 'train.txt']
# 遍历每个文件并追加内容
for file_name in files_to_process:
    source_file_path = os.path.join(source_dir, file_name)
    target_file_path = os.path.join(target_dir, file_name)
    # 检查源文件是否存在
    if not os.path.exists(source_file_path):
        print(f"Source file {source_file_path} does not exist. Skipping.")
        continue
    # 读取源文件内容
    with open(source_file_path, 'r') as source_file:
        source_content = source_file.read()
    # Check if the last character of source_content is a newline
    if not source_content.endswith('\n'):
        source_content += '\n'
    
    # Append content to the target file
    with open(target_file_path, 'a') as target_file:
        target_file.write(source_content)
    print(f"Content from {source_file_path} has been appended to {target_file_path}")
print("All operations completed.")