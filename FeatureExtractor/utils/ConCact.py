import argparse
import numpy as np


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Combine text and image features')
    parser.add_argument('--text-file', required=True, help='Path to the text feature file')
    parser.add_argument('--image-file', required=True, help='Path to the image feature file')
    parser.add_argument('--output-file', required=True, help='Path to the output combined feature file')
    args = parser.parse_args()

    # 加载文本和图像特征文件
    text_feats = np.load(args.text_file)
    image_feats = np.load(args.image_file)

    # 合并特征文件
    combined_features = np.concatenate((text_feats, image_feats), axis=1)

    # 保存新特征文件
    np.save(args.output_file, combined_features)

    print(f'Combined features saved to: {args.output_file}')


if __name__ == '__main__':
    main()
