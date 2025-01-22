import os
from PIL import Image
import sys


def get_image_resolutions(image_folder):
    resolutions = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            file_path = os.path.join(image_folder, filename)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    resolutions.append((width, height))
            except (IOError, OSError, Image.UnidentifiedImageError):
                print(f'无法识别的图像文件: {filename}')

    return resolutions


def calculate_statistics(resolutions):
    if not resolutions:
        print("没有可用的图像数据进行统计。")
        return None

    widths = [res[0] for res in resolutions]
    heights = [res[1] for res in resolutions]

    min_width, min_height = min(widths), min(heights)
    max_width, max_height = max(widths), max(heights)
    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)

    return {
        "min_resolution": (min_width, min_height),
        "max_resolution": (max_width, max_height),
        "avg_resolution": (avg_width, avg_height),
        "total_images": len(resolutions)
    }


def display_statistics(statistics):
    if statistics:
        print(f"总图像数量: {statistics['total_images']}")
        print(f"最小分辨率: {statistics['min_resolution'][0]}x{statistics['min_resolution'][1]}")
        print(f"最大分辨率: {statistics['max_resolution'][0]}x{statistics['max_resolution'][1]}")
        print(f"平均分辨率: {statistics['avg_resolution'][0]:.2f}x{statistics['avg_resolution'][1]:.2f}")


def main(image_folder):
    resolutions = get_image_resolutions(image_folder)
    statistics = calculate_statistics(resolutions)
    display_statistics(statistics)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
        if os.path.isdir(image_folder):
            main(image_folder)
        else:
            print("无效的文件夹路径，请提供一个有效的文件夹路径。")
    else:
        print("请提供一个图像文件夹路径作为命令行参数。")
