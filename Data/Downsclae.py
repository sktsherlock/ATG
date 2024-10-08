import os
import cv2
import numpy as np
from PIL import Image
import argparse


def calculate_target_size(input_path):
    """
    计算目标图像大小,即原始图像大小的一半

    参数:
    input_path (str): 输入图像的路径

    返回:
    target_size (tuple): 目标图像尺寸(宽, 高)
    """
    try:
        # 尝试读取图像
        image = cv2.imread(input_path)

        # 检查图像是否成功读取
        if image is not None:
            original_height, original_width, _ = image.shape
            target_width = original_width // 2
            target_height = original_height // 2
            target_size = (target_width, target_height)
            return target_size
        else:
            # 如果图像无法读取,返回 None
            return None
    except (cv2.error, AttributeError):
        # 捕获可能的异常并返回 None
        return None


def downscale_image(input_path, output_path):
    """
    将高分辨率图像下采样成低分辨率图像,目标图像大小为原始图像的一半

    参数:
    input_path (str): 输入图像的路径
    output_path (str): 输出图像的路径
    """
    try:
        # 1. 计算目标图像大小
        target_size = calculate_target_size(input_path)
        if target_size is None:
            return

        # 2. 读取输入图像
        image = cv2.imread(input_path)

        # 3. 调整图像大小
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # 4. 保存下采样后的图像
        cv2.imwrite(output_path, resized_image)
    except (cv2.error, IOError) as e:
        print(f"Error processing image {input_path}: {e}")


def process_images(input_dir, output_dir):
    """
    处理输入目录中的所有图像,并将下采样后的图像保存到输出目录

    参数:
    input_dir (str): 输入图像的目录路径
    output_dir (str): 输出图像的目录路径
    """
    # 创建输出目录(如果不存在)
    os.makedirs(output_dir, exist_ok=True)

    # 添加计数器变量
    processed_count = 0
    # 遍历输入目录中的所有图像文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 调用下采样函数
            downscale_image(input_path, output_path)
            # 增加计数器
            processed_count += 1

            # 每处理500个图像后输出一次消息
            if processed_count % 500 == 0:
                print(f'已将前 {processed_count} 个图像下采样至原始图像一半大小并保存')


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="将输入目录中的图像下采样至原始图像一半大小")
    parser.add_argument("--input_dir", required=True, help="输入图像的目录路径")
    parser.add_argument("--output_dir", required=True, help="输出图像的目录路径")
    args = parser.parse_args()

    # 调用 process_images 函数
    process_images(args.input_dir, args.output_dir)