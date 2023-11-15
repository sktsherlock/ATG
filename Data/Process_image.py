import os
import shutil
import pandas as pd

# 读取CSV文件
csv_file = 'magazines.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 创建新的数据集文件夹
dataset_folder = 'Amazon-Magazine'  # 替换为你想要创建的数据集文件夹路径
os.makedirs(dataset_folder, exist_ok=True)

# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    id_value = row['id']
    label_value = row['label']

    # 根据id创建类别文件夹
    class_folder = os.path.join(dataset_folder, str(label_value))
    os.makedirs(class_folder, exist_ok=True)

    # 拷贝相应的图像文件到类别文件夹
    image_file = f'magazineImages/{id_value}-1.jpg'  # 替换为你的图像文件夹路径和命名规则
    if os.path.exists(image_file):
        dest_file = os.path.join(class_folder, f'{id_value}-1.jpg')
        shutil.copy(image_file, dest_file)
    else:
        print(f"Image file not found for id: {id_value}")

print("Dataset creation completed.")
