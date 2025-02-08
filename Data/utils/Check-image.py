import os
from PIL import Image


def check_image_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # 尝试打开图像文件
                with Image.open(file_path) as img:
                    img.verify()  # 验证图像文件是否完整
            except (IOError, SyntaxError) as e:
                print(f"Error reading image file: {file_path}")
                print(f"Error message: {str(e)}")
            except Image.DecompressionBombError as e:
                print(f"Decompression Bomb Error: {file_path}")
                print(f"Error message: {str(e)}")

# 检查train文件夹下的图像文件
train_folder = "Amazon-Magazine/train"
print("Checking train folder...")
for label_folder in os.listdir(train_folder):
    label_folder_path = os.path.join(train_folder, label_folder)
    if os.path.isdir(label_folder_path):
        check_image_files(label_folder_path)

# 检查val文件夹下的图像文件
val_folder = "Amazon-Magazine/val"
print("Checking val folder...")
for label_folder in os.listdir(val_folder):
    label_folder_path = os.path.join(val_folder, label_folder)
    if os.path.isdir(label_folder_path):
        check_image_files(label_folder_path)

# 检查test文件夹下的图像文件
test_folder = "Amazon-Magazine/test"
print("Checking test folder...")
for label_folder in os.listdir(test_folder):
    label_folder_path = os.path.join(test_folder, label_folder)
    if os.path.isdir(label_folder_path):
        check_image_files(label_folder_path)