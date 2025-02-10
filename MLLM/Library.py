import torch
from transformers import AutoProcessor

# 这里需要显式导入特定的模型
from transformers import (
    MllamaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
)


def load_model_and_processor(model_name: str):
    """
    根据 Hugging Face 的模型名称加载对应的模型和处理器。
    """
    model_mapping = {
        "meta-llama/Llama-3.2-11B-Vision-Instruct": {
            "model_cls": MllamaForConditionalGeneration,
            "processor_cls": AutoProcessor,
        },
        "Qwen/Qwen2.5-VL-7B-Instruct": {
            "model_cls": Qwen2_5_VLForConditionalGeneration,
            "processor_cls": AutoProcessor,
        },
        "Qwen/Qwen2-VL-7B-Instruct": {
            "model_cls": Qwen2VLForConditionalGeneration,
            "processor_cls": AutoProcessor,
        },
        "llava-hf/llava-onevision-qwen2-7b-ov-hf": {
            "model_cls": LlavaOnevisionForConditionalGeneration,
            "processor_cls": AutoProcessor,
        },
        "google/paligemma2-3b-pt-448": {
            "model_cls": PaliGemmaForConditionalGeneration,
            "processor_cls": AutoProcessor,
        },
        "google/paligemma2-3b-pt-896": {
            "model_cls": PaliGemmaForConditionalGeneration,
            "processor_cls": AutoProcessor,
        },
    }

    if model_name not in model_mapping:
        raise ValueError(f"不支持的模型名称: {model_name}")

    model_cls = model_mapping[model_name]["model_cls"]
    processor_cls = model_mapping[model_name]["processor_cls"]

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = processor_cls.from_pretrained(model_name)

    return model, processor


# 示例使用
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # 直接使用 Hugging Face 上的模型 ID
model, processor = load_model_and_processor(model_name)

print(f"成功加载模型: {model_name}")
