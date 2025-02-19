import os
import numpy as np


def load_npy_files(folder_path):
    # 检查提供的路径是否为有效的目录
    if not os.path.isdir(folder_path):
        print(f"提供的路径 '{folder_path}' 不是一个有效的目录。")
        return

    # 遍历目录及其子目录
    for root, _, files in os.walk(folder_path):
        for file in files:
            # 筛选出以 .npy 结尾的文件
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    # 加载 .npy 文件
                    data = np.load(file_path)
                    # 打印文件名和数据的形状
                    print(f"文件名: {file}")
                    print(f"特征维度: {data.shape}")
                except Exception as e:
                    print(f"无法加载文件 '{file_path}'。错误: {e}")


# 示例用法
text_folder_path = '/home/aiscuser/ATG/Data/Movies/TextFeature'  # 替换为您的文件夹路径
visual_folder_path = '/home/aiscuser/ATG/Data/Movies/ImageFeature'  # 替换为您的文件夹路径
multimodal_folder_path = '/home/aiscuser/ATG/Data/Movies/MMFeature'  # 替换为您的文件夹路径

load_npy_files(text_folder_path)
load_npy_files(visual_folder_path)
load_npy_files(multimodal_folder_path)
