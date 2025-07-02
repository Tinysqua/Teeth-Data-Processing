import os
import shutil
from tqdm import tqdm

def organize_dental_files(root_dir):
    # 创建目标文件夹
    save_dir = '/data1/shizhen/TeethZip/Aidite_Crown_Dataset_sixth'
    tooth_crown_dir = os.path.join(save_dir, 'tooth_crown')
    jaw_dir = os.path.join(save_dir, 'jaw')
    normal_info = os.path.join(save_dir, 'normal_info')
    
    
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
        crown_files = [f for f in files if f.startswith('tooth_crown') and f.endswith('.ply')]
        if len(crown_files) > 1:
            print(f"文件夹 {folder} 中发现多个牙冠文件: {crown_files}")
            
        # 提取所有牙齿编号
        tooth_numbers = []
        for crown_file in crown_files:
            # 提取牙齿编号
            tooth_number = crown_file.split('_')[-1].split('.')[0]
            tooth_numbers.append(tooth_number)
            
            # 新的文件名格式：数字_牙齿编号.ply
            new_crown_name = f"{int(folder)}_{tooth_number}.ply"
            
            # 复制文件
            src_path = os.path.join(folder_path, crown_file)
            dst_path = os.path.join(tooth_crown_dir, new_crown_name)
            shutil.copy2(src_path, dst_path)

        # 处理上颌和下颌文件
        for jaw_type in ['upper', 'lower']:
            # 根据类型获取文件列表
            jaw_files = [f for f in files if f'{jaw_type}_jaw' in f.lower() and f.endswith('.ply')]
            
            if len(jaw_files) > 1:
                print(f"文件夹 {folder} 中发现多个{jaw_type}颌文件: {jaw_files}")
            elif len(jaw_files) == 1 and len(crown_files) > 1:
                print(f"文件夹 {folder} 中有多个牙冠文件但只有一个{jaw_type}颌文件，将为每个牙冠复制对应的{jaw_type}颌文件")
                
                # 只有一个颌文件但有多个牙冠
                jaw_file = jaw_files[0]
                src_path = os.path.join(folder_path, jaw_file)
                
                # 为每个牙齿编号复制一次颌文件
                for tooth_number in tooth_numbers:
                    new_jaw_name = f"{jaw_type}_{int(folder)}_{tooth_number}.ply"
                    dst_path = os.path.join(jaw_dir, new_jaw_name)
                    shutil.copy2(src_path, dst_path)
            else:
                # 正常情况，按照原逻辑处理
                for jaw_file in jaw_files:
                    # 如果有牙冠文件，使用第一个牙冠文件的编号
                    tooth_number = tooth_numbers[0] if tooth_numbers else ''
                    new_jaw_name = f"{jaw_type}_{int(folder)}_{tooth_number}.ply"
                    
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
    root_directory = "/data1/shizhen/TeethZip/sixth_zip"
    organize_dental_files(root_directory)
