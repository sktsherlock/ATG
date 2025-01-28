import os
import ast
import torch
import pandas as pd
from PIL import Image
from dgl import load_graphs
from transformers import MllamaForConditionalGeneration, AutoProcessor


# 参数配置
class Config:
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # 修改路径解析逻辑
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录：/home/aiscuser/ATG/MLLM
    data_root = os.path.normpath(os.path.join(current_dir, "../Data/Movies"))  # 关键修改点：只需上一级到ATG目录
    classes = ["action", "comedy", "drama"]
    max_new_tokens = 15
    image_ext = ".jpg"

# 其他代码保持不变...

# 加载数据集函数修改路径验证
def load_movie_dataset():
    """加载电影数据集"""
    # 验证路径是否存在
    if not os.path.exists(Config.data_root):
        raise FileNotFoundError(
            f"Dataset path not found: {Config.data_root}\n"
            f"Current working directory: {os.getcwd()}"
        )

    csv_path = os.path.join(Config.data_root, "Movies.csv")
    graph_path = os.path.join(Config.data_root, "MoviesGraph.pt")

    # 验证文件存在性
    for path in [csv_path, graph_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {path}")

    df = pd.read_csv(csv_path, converters={'neighbors': ast.literal_eval})
    graph = load_graphs(graph_path)[0][0]

    return df, graph


# 图像加载器修改
class MovieImageLoader:
    def __init__(self):
        self.image_dir = os.path.join(Config.data_root, "MoviesImages")
        # 验证图像目录存在性
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory missing: {self.image_dir}")


# 辅助函数：构建分类提示
def build_classification_prompt(text: str, classes: list) -> list:
    """构建适合分类任务的prompt模板"""
    return [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"""
                As a movie expert, classify this item based on its visual and textual features.
                Text description: {text}
                Available categories: {", ".join(classes)}
                Answer ONLY with the category name.
            """}
        ]}
    ]


# 加载模型
model = MllamaForConditionalGeneration.from_pretrained(
    Config.model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(Config.model_id)


# 评估函数
def run_evaluation(dataframe, num_samples=5):
    image_loader = MovieImageLoader()
    correct = 0

    for idx, row in dataframe.iterrows():
        if idx >= num_samples:
            break

        try:
            node_id = row["node_id"]
            text = row["text"]
            label = row["label"].lower()  # 统一转为小写

            # 加载图像
            image = image_loader.load_image(node_id)

            # 构建prompt
            messages = build_classification_prompt(text, Config.classes)
            input_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            # 处理输入
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            # 生成结果
            output = model.generate(**inputs, max_new_tokens=Config.max_new_tokens)
            prediction = processor.decode(output[0], skip_special_tokens=True).strip().lower()

            # 解析结果（取第一个匹配的类别）
            predicted_class = next((c for c in Config.classes if c in prediction), None)

            # 计算准确率
            if predicted_class == label:
                correct += 1

            print(f"Node {node_id}:")
            print(f"Predicted: {predicted_class} | Ground Truth: {label}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing node {node_id}: {str(e)}")
            continue

    print(f"\nAccuracy: {correct / num_samples:.2f} ({correct}/{num_samples})")


if __name__ == "__main__":
    # 加载数据
    df, graph = load_movie_dataset()

    # 运行评估（使用前5个样本）
    run_evaluation(df, num_samples=5)
