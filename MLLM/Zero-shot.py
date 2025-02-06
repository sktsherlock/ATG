import os
import ast
import argparse
import torch
import time
import pandas as pd
from PIL import Image
import numpy as np
import dgl
from dgl import load_graphs
import networkx as nx
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import MllamaForConditionalGeneration, AutoProcessor


def split_dataset(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)

    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]

    return train_ids, val_ids, test_ids



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multimodal Node Classification with MLLM and RAG-enhanced inference')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-11B-Vision-Instruct',
                        help='HuggingFace模型名称或路径')
    parser.add_argument('--dataset_name', type=str, default='Movies',
                        help='数据集名称（对应Data目录下的子目录名）')
    parser.add_argument('--base_dir', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help='项目根目录路径')
    parser.add_argument('--label_column', type=str, default='label',
                        help='CSV文件中表示数字化标签的列名')
    parser.add_argument('--text_label_column', type=str, default='second_category',
                        help='CSV文件中表示文本类别标签的列名')
    parser.add_argument('--max_new_tokens', type=int, default=15,
                        help='生成的最大token数量')
    parser.add_argument('--image_ext', type=str, default='.jpg',
                        help='图像文件扩展名')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='测试样本数量')
    parser.add_argument('--k_hop', type=int, default=0,
                        help='RAG增强推理时使用的邻居阶数（0表示不使用邻居）')
    parser.add_argument(
        "--train_ratio", type=float, default=0.6, help="training ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="training ratio"
    )
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

        # 检查标签列是否存在
        if self.args.label_column not in df.columns:
            raise ValueError(f"Label column '{self.args.label_column}' not found in CSV file.")
        if self.args.text_label_column not in df.columns:
            raise ValueError(f"Text label column '{self.args.text_label_column}' not found in CSV file.")

        # 加载图数据（DGL格式）
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


def get_k_hop_neighbors(nx_graph, node_id, k):
    """
    获取指定中心节点的 k-hop 邻居（不包含中心节点）
    使用 NetworkX 的 single_source_shortest_path_length 方法。
    """
    neighbors = set()
    for target, distance in nx.single_source_shortest_path_length(nx_graph, node_id).items():
        if 0 < distance <= k:
            neighbors.add(target)
    return list(neighbors)


def build_classification_prompt_with_neighbors(center_text: str, neighbor_texts: list, classes: list) -> str:
    """
    Build a RAG-enhanced classification prompt by integrating the center node's text with its neighbors' information.
    """
    prompt = f"Center node description: {center_text}\n"
    if neighbor_texts:
        prompt += "Below are the descriptions of related neighbor nodes:\n"
        for idx, n_text in enumerate(neighbor_texts):
            prompt += f"Neighbor {idx+1} description: {n_text}\n"
    prompt += f"Based on the above information, classify this node into one of the following categories: {', '.join(classes)}.\n" \
              "Please answer with the category name ONLY."
    return prompt.strip()


def build_classification_prompt(center_text: str, classes: list) -> str:
    """Build a basic classification prompt without neighbor information."""
    prompt = (
        f"Available categories: {', '.join(classes)}.\n"
        f"Description: {center_text}\n"
        "Based on the multimodal information above, please choose the most appropriate category.\n"
        "Do not simply choose the first category; analyze the description carefully and consider all available options.\n"
        "Answer ONLY with the exact category name."
    )
    return prompt.strip()



def main(args):
    start_time = time.time()  # 记录起始时间
    # 初始化数据加载器
    dataset_loader = DatasetLoader(args)
    df, dgl_graph = dataset_loader.load_data()

    # 从CSV中提取所有唯一类别，并排序（可根据需要调整顺序）
    classes = sorted(df[args.text_label_column].str.lower().unique())

    # 构建一个从节点ID到节点数据的字典，便于后续查找邻居信息
    # 假设 CSV 中 "id" 列作为唯一标识符，且 "text" 为节点描述
    node_data_dict = {row["id"]: row for _, row in df.iterrows()}

    # 加载模型和处理器
    model = MllamaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # 如果使用 RAG 增强推理，转换 DGL 图为 NetworkX 图
    if args.k_hop > 0:
        nx_graph = dgl.to_networkx(dgl_graph, node_attrs=['_ID'])  # 根据实际情况设置节点属性
    else:
        nx_graph = None

    # 初始化计数器
    y_true = []
    y_pred = []
    total_samples = 0
    mismatch_count = 0  # 统计预测类别完全不匹配的情况

    # 进行数据集划分
    train_ids, val_ids, test_ids = split_dataset(
        nodes_num=len(df),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    selected_ids = test_ids  # 这里可以选择 train_ids, val_ids, 或 test_ids
    sample_df = df.iloc[selected_ids]  # 使用 selected_ids 来选择相应的数据集
    # 从所选的子集数据中，再选择前 num_samples 个样本
    sample_df = sample_df.head(args.num_samples)  # 选择前 num_samples 个样本
    for idx, row in tqdm(sample_df.iterrows(), total=sample_df.shape[0], desc="Processing samples"):
        try:
            node_id = row["id"]
            text = row["text"]
            numeric_label = row[args.label_column]  # 数字化标签
            text_label = row[args.text_label_column].lower()  # 文本类别标签

            # 加载图像
            image = dataset_loader.load_image(node_id)

            # 构建提示
            if args.k_hop > 0 and nx_graph is not None:
                # 获取 k-hop 邻居节点 ID
                neighbor_ids = get_k_hop_neighbors(nx_graph, node_id, args.k_hop)
                # 从字典中提取邻居的文本描述（若存在）
                neighbor_texts = []
                for nid in neighbor_ids:
                    if nid in node_data_dict:
                        neighbor_texts.append(str(node_data_dict[nid].get("text", "")))
                prompt_text = build_classification_prompt_with_neighbors(text, neighbor_texts, classes)
            else:
                # 使用基本提示，不进行邻居增强
                prompt_text = build_classification_prompt(text, classes)

            # # 使用处理器生成输入文本（支持多模态Chat模板）
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=False)


            # 处理图像和文本输入
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            # 打印输入的图像和文本信息以进行调试
            print("Input Image:", image)
            print("Input Text:", input_text)
            # 生成预测结果
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, temperature=1.0, top_k=50, top_p=0.95)
            output_tokens = output[0][len(inputs["input_ids"][0]):]
            prediction = processor.decode(output_tokens, skip_special_tokens=True).strip().lower()
            # prediction = processor.decode(output[0], skip_special_tokens=True).strip().lower()

            # 简单解析预测结果，匹配类别列表中的关键词
            print("Prediction:", prediction)
            predicted_class = next((c for c in classes if c in prediction), None)
            print("Predicted Class:", predicted_class)

            total_samples +=1
            # 计算完全不匹配的情况
            if predicted_class is None:  # 预测结果与类别列表完全不匹配
                mismatch_count += 1

            # 收集真实值和预测值
            y_true.append(text_label)
            y_pred.append(predicted_class if predicted_class else "unknown")  # 用 "unknown" 代替未匹配的类别

            print(f"Node {node_id}:")
            print("Prompt:")
            print(prompt_text)
            print(f"Predicted: {predicted_class} | GT: {text_label} (Numeric: {numeric_label})")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing node {node_id}: {str(e)}")

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算 Macro-F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 计算不匹配概率
    mismatch_probability = mismatch_count / total_samples

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Mismatch Probability: {mismatch_probability:.4f}")

    # 记录结束时间并计算耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time spent: {total_time:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)
