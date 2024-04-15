import os
from PIL import Image



def check_image_files(image_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            file_path = os.path.join(image_folder, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    if width * height > 89478485:
                        print('Image size超过限制的文件名:', filename)
            except (IOError, OSError, Image.UnidentifiedImageError):
                print('无法识别的图像文件:', filename)

# 指定图像文件夹路径
image_folder = 'Reddit/RedditImages/'
check_image_files(image_folder)