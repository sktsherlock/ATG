import numpy as np
import os
import argparse


def process_nan_inf(file_path, strategy="mean", save=True, output_path=None):
    """
    处理 npy 特征文件中的 NaN 和 Inf 值。

    参数：
    - file_path: str，要处理的 .npy 文件路径
    - strategy: str，填充策略，可选 ["mean", "median", "zero"]
    - save: bool，是否保存处理后的数据
    - output_path: str，处理后数据的保存路径，默认覆盖原文件

    返回：
    - processed_data: np.ndarray，处理后的特征数据
    """
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return None

    try:
        # 加载数据
        data = np.load(file_path)

        # 检查数据是否有 NaN 或 Inf
        nan_mask = np.isnan(data)
        inf_mask = np.isinf(data)

        if not nan_mask.any() and not inf_mask.any():
            print(f"{file_path} 没有 NaN 或 Inf，无需处理。")
            return data  # 无需修改，直接返回

        print(f"发现 NaN: {nan_mask.sum()} 个, Inf: {inf_mask.sum()} 个")

        # 处理 NaN 和 Inf
        if strategy == "mean":
            col_means = np.nanmean(data, axis=0)
            col_means[np.isinf(col_means)] = 0
            data[nan_mask | inf_mask] = np.take(col_means, np.where(nan_mask | inf_mask)[1])

        elif strategy == "median":
            col_medians = np.nanmedian(data, axis=0)
            col_medians[np.isinf(col_medians)] = 0
            data[nan_mask | inf_mask] = np.take(col_medians, np.where(nan_mask | inf_mask)[1])

        elif strategy == "zero":
            data[nan_mask | inf_mask] = 0

        else:
            raise ValueError("不支持的填充策略！可选: 'mean', 'median', 'zero'")

        print(f"处理完成，已使用 '{strategy}' 策略填充 NaN 和 Inf")

        # 保存数据
        if save:
            output_file = output_path if output_path else file_path
            np.save(output_file, data)
            print(f"处理后的数据已保存至: {output_file}")

        return data

    except Exception as e:
        print(f"处理 {file_path} 时发生错误: {e}")
        return None


def process_directory(directory, strategy="mean", save=True):
    """
    处理目录下所有的 .npy 文件。

    参数：
    - directory: str，要处理的目录路径
    - strategy: str，填充策略，可选 ["mean", "median", "zero"]
    - save: bool，是否保存处理后的数据
    """
    if not os.path.isdir(directory):
        print(f"目录 {directory} 不存在！")
        return

    npy_files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    if not npy_files:
        print(f"目录 {directory} 中没有 .npy 文件！")
        return

    print(f" 在 {directory} 目录下找到 {len(npy_files)} 个 .npy 文件，开始处理...")
    for npy_file in npy_files:
        file_path = os.path.join(directory, npy_file)
        process_nan_inf(file_path, strategy=strategy, save=save)
    print("目录处理完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 npy 特征文件中的 NaN 和 Inf")
    parser.add_argument("path", type=str, help="要处理的文件或目录路径")
    parser.add_argument("--strategy", type=str, choices=["mean", "median", "zero"], default="mean",
                        help="填充策略（默认: mean）")
    parser.add_argument("--save", action="store_true", help="是否保存处理后的数据（默认不保存）")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        process_directory(args.path, strategy=args.strategy, save=args.save)
    elif os.path.isfile(args.path) and args.path.endswith(".npy"):
        process_nan_inf(args.path, strategy=args.strategy, save=args.save)
    else:
        print("请输入正确的 .npy 文件路径或目录路径！")
