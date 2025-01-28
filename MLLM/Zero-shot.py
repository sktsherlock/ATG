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
    data_root = "Data/Movies"  # 数据集根目录
    classes = ["action", "comedy", "drama"]  # 替换为真实类别
    max_new_tokens = 15
    image_ext = ".jpg"  # 图像文件扩展名


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


# 加载数据集
def load_movie_dataset():
    """加载电影数据集"""
    # 读取节点信息
    csv_path = os.path.join(Config.data_root, "Movies.csv")
    df = pd.read_csv(csv_path, converters={
        'neighbors': ast.literal_eval  # 将字符串形式的列表转换为实际列表
    })

    # 读取图数据（备用）
    graph_path = os.path.join(Config.data_root, "MoviesGraph.pt")
    graph = load_graphs(graph_path)[0][0]  # 加载DGL图

    return df, graph


# 图像加载器
class MovieImageLoader:
    def __init__(self):
        self.image_dir = os.path.join(Config.data_root, "MoviesImages")

    def load_image(self, node_id: int) -> Image.Image:
        """根据节点ID加载图像"""
        img_path = os.path.join(
            self.image_dir,
            f"{node_id}{Config.image_ext}"
        )
        try:
            return Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            raise ValueError(f"Image not found for node {node_id} at {img_path}")


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