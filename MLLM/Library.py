import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
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
    # 如果是Qwen系列，在加入max_pixels = 1280 * 28 * 28
    if "Qwen" in model_name:
        processor = processor_cls.from_pretrained(model_name, max_pixels=1280 * 28 * 28)
    else:
        processor = processor_cls.from_pretrained(model_name)

    return model, processor


def prepare_inputs_for_model(messages, input_text, images, center_image, processor, model, args, name):
    if name in ["Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
        # 处理图像或视频输入（Qwen 需要使用 `process_vision_info()`）
        image_inputs, video_inputs = process_vision_info(messages)

        # Qwen 多模态处理
        inputs = processor(
            text=[input_text],
            images=image_inputs,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)
    else:
        # 其他模型（如 LLaVA）
        inputs = processor(
            images if args.neighbor_mode in ["image", "both"] and args.num_neighbours > 0 else center_image,  # 选择输入的图片
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

    return inputs


if __name__ == "__main__":
    # 示例使用
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"  # 直接使用 Hugging Face 上的模型 ID
    Model, Processor = load_model_and_processor(model_name)

    print(f"成功加载模型: {model_name}")
