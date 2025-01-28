import os
import ast
import argparse
import torch
import pandas as pd
from PIL import Image
from dgl import load_graphs
from transformers import MllamaForConditionalGeneration, AutoProcessor


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multimodal Node Classification with MLLM')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct',
                        help='HuggingFace模型名称或路径')
    parser.add_argument('--dataset_name', type=str, default='Movies',
                        help='数据集名称（对应Data目录下的子目录名）')
    parser.add_argument('--base_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help='项目根目录路径')
    parser.add_argument('--classes', nargs='+', default=["action", "comedy", "drama"],
                        help='分类类别列表')
    parser.add_argument('--max_new_tokens', type=int, default=15,
                        help='生成的最大token数量')
    parser.add_argument('--image_ext', type=str, default='.jpg',
                        help='图像文件扩展名')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='测试样本数量')
    return parser.parse_args()


class DatasetLoader:
    def __init__(self, args):
        """初始化数据集加载器"""
        self.args = args
        self.data_dir = os.path.join(args.base_dir, 'Data', args.dataset_name)
        self._verify_paths()

    def _verify_paths(self):
        """验证必要路径是否存在"""
        required_files = [
            os.path.join(self.data_dir, f"{self.args.dataset_name}.csv"),
            os.path.join(self.data_dir, f"{self.args.dataset_name}Graph.pt"),
            os.path.join(self.data_dir, f"{self.args.dataset_name}Images")
        ]

        missing = [path for path in required_files if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Missing required files/directories: {missing}")

    def load_data(self):
        """加载数据集"""
        # 加载CSV数据
        csv_path = os.path.join(self.data_dir, f"{self.args.dataset_name}.csv")
        df = pd.read_csv(csv_path, converters={'neighbors': ast.literal_eval})

        # 加载图数据
        graph_path = os.path.join(self.data_dir, f"{self.args.dataset_name}Graph.pt")
        graph = load_graphs(graph_path)[0][0]

        return df, graph

    def load_image(self, node_id: int) -> Image.Image:
        """加载节点图像"""
        img_path = os.path.join(
            self.data_dir,
            f"{self.args.dataset_name}Images",
            f"{node_id}{self.args.image_ext}"
        )
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")


def build_classification_prompt(text: str, classes: list) -> list:
    """构建分类提示模板"""
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"""
                Based on the multimodal information, classify this node into one of: {", ".join(classes)}.
                Text description: {text}
                Answer ONLY with the class name.
            """}
        ]}
    ]


def main(args):
    # 初始化组件
    dataset_loader = DatasetLoader(args)
    df, graph = dataset_loader.load_data()

    # 加载模型
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # 运行评估
    correct = 0
    for idx, row in df.head(args.num_samples).iterrows():
        try:
            node_id = row["id"]
            text = row["text"]
            label = row["second_category"].lower()

            # 准备输入
            image = dataset_loader.load_image(node_id)
            messages = build_classification_prompt(text, args.classes)
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            # 生成预测
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            prediction = processor.decode(output[0], skip_special_tokens=True).strip().lower()

            # 解析结果
            predicted_class = next((c for c in args.classes if c in prediction), None)
            if predicted_class == label:
                correct += 1

            print(f"Node {node_id}: {predicted_class} | GT: {label}")

        except Exception as e:
            print(f"Error processing node {node_id}: {str(e)}")

    print(f"\nFinal Accuracy: {correct / args.num_samples:.2f} ({correct}/{args.num_samples})")


if __name__ == "__main__":
    args = parse_args()
    main(args)