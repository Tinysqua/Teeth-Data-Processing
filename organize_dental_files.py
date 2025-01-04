import os
import shutil
from tqdm import tqdm

def organize_dental_files(root_dir):
    # 创建目标文件夹
    tooth_crown_dir = '/xxx/tooth_crown'
    jaw_dir = '/xxx/jaw'
    normal_info = '/xxx/normal_info'
    
    # 如果目标文件夹不存在，则创建
    os.makedirs(tooth_crown_dir, exist_ok=True)
    os.makedirs(jaw_dir, exist_ok=True)
    os.makedirs(normal_info, exist_ok=True)

    # 获取所有数字文件夹
    folders = [f for f in os.listdir(root_dir) if f.isdigit()]
    folders = sorted(folders, key=int)
    
    for folder in tqdm(folders, desc='Processing folders', total=len(folders)):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # 获取文件夹中的所有文件
        files = os.listdir(folder_path)
        
        # 处理牙冠文件
        crown_files = [f for f in files if f.startswith('tooth_crown')]
        for crown_file in crown_files:
            # 提取牙齿编号
            tooth_number = crown_file.split('_')[-1].split('.')[0]
            # 新的文件名格式：数字_牙齿编号.ply
            new_crown_name = f"{int(folder)}_{tooth_number}.ply"
            
            # 复制文件
            src_path = os.path.join(folder_path, crown_file)
            dst_path = os.path.join(tooth_crown_dir, new_crown_name)
            shutil.copy2(src_path, dst_path)

        # 处理上颌文件
        upper_jaw_files = [f for f in files if 'upper' in f.lower()]
        for jaw_file in upper_jaw_files:
            # 提取牙齿编号（如果有的话，这里假设使用crown文件的编号）
            tooth_number = crown_files[0].split('_')[-1].split('.')[0] if crown_files else ''
            new_jaw_name = f"upper_{int(folder)}_{tooth_number}.ply"
            
            src_path = os.path.join(folder_path, jaw_file)
            dst_path = os.path.join(jaw_dir, new_jaw_name)
            shutil.copy2(src_path, dst_path)

        # 处理下颌文件
        lower_jaw_files = [f for f in files if 'lower' in f.lower()]
        for jaw_file in lower_jaw_files:
            # 提取牙齿编号（如果有的话，这里假设使用crown文件的编号）
            tooth_number = crown_files[0].split('_')[-1].split('.')[0] if crown_files else ''
            new_jaw_name = f"lower_{int(folder)}_{tooth_number}.ply"
            
            src_path = os.path.join(folder_path, jaw_file)
            dst_path = os.path.join(jaw_dir, new_jaw_name)
            shutil.copy2(src_path, dst_path)

        # 处理normal info文件
        json_files = [f for f in files if f.endswith('.json')]
        for json_file in json_files:
            new_json_name = f"{int(folder)}_{json_file}"
            
            src_path = os.path.join(folder_path, json_file)
            dst_path = os.path.join(normal_info, new_json_name)
            shutil.copy2(src_path, dst_path)


# 使用示例
if __name__ == "__main__":
    # 替换为你的根目录路径
    root_directory = "/ssd_data/shizhen/aidite_dataset1205"
    organize_dental_files(root_directory)
