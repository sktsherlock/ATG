import os
from PIL import Image
import sys


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


def check_image_resolution(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f'Image resolution: {width}x{height}')
    except (IOError, OSError, Image.UnidentifiedImageError):
        print('无法识别的图像文件:', image_path)


# 从命令行参数获取图像文件夹路径
if len(sys.argv) > 1:
    path = sys.argv[1]
    if os.path.isdir(path):
        check_image_files(path)
    elif os.path.isfile(path):
        check_image_resolution(path)
    else:
        print('无效的路径:', path)
else:
    print('请提供一个图像文件路径或图像文件夹路径作为命令行参数。')

# image_folder = 'RedditS/RedditSImages/'