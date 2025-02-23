# Movies数据集
#      - Data/Movies/MMFeature/Movies_Qwen2-VL-7B-Instruct_tv.npy
#      - Data/Movies/MMFeature/Movies_Llama-3.2-11B-Vision-Instruct_tv.npy
#      - Data/Movies/MMFeature/Movies_LLAMA8B_CLIP.npy
#      - Data/Movies/TextFeature/Movies_roberta_base_512_mean.npy
#      - Data/Movies/TextFeature/Movies_Llama_3.2_1B_Instruct_512_mean.npy
#      - Data/Movies/TextFeature/Movies_Llama_3.1_8B_Instruct_512_mean.npy
#      - Data/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy
#      - Data/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy
#      - Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy
#      - Data/Movies/ImageFeature/Movies_swinv2_large.npy
#      - Data/Movies/ImageFeature/Movies_convnextv2_huge.npy
#      - Data/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy
#      - Data/Movies/ImageFeature/Movies_Qwen2-VL-7B-Instruct_visual.npy

python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/MMFeature/Movies_Qwen2-VL-7B-Instruct_tv.npy --label1 Qwen2-VL-7B   --feat2 /home/aiscuser/ATG/Data/Movies/MMFeature/Movies_Llama-3.2-11B-Vision-Instruct_tv.npy --label2 Llama3.2-VL-11B --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/MMFeature/Movies_LLAMA8B_CLIP.npy --label1 LLaMA+CLIP  --feat2 /home/aiscuser/ATG/Data/Movies/TextFeature/Movies_roberta_base_512_mean.npy --label2 RoBERTa  --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/TextFeature/Movies_Llama_3.2_1B_Instruct_512_mean.npy --label1 Llama3.2-1B --feat2 /home/aiscuser/ATG/Data/Movies/TextFeature/Movies_Llama_3.1_8B_Instruct_512_mean.npy --label2 Llama3.1-8B  --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy --label1 Llama3.2-VL-T --feat2 /home/aiscuser/ATG/Data/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy --label2 Qwen2-VL-T  --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --label1 CLIP --feat2 /home/aiscuser/ATG/Data/Movies/ImageFeature/Movies_swinv2_large.npy --label2 SwinV2 --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/ImageFeature/Movies_convnextv2_huge.npy --label1 ConvNeXT --feat2 /home/aiscuser/ATG/Data/Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy --label2 Llama3.2-VL-V  --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Movies/ImageFeature/Movies_Qwen2-VL-7B-Instruct_visual.npy --label1 Qwen2-VL-V --feat2 /home/aiscuser/ATG/Data/Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy --label2 Qwen2-VL-T  --dataname Movies --graph_path /home/aiscuser/ATG/Data/Movies/MoviesGraph.pt

# Toys 数据集
#      - Data/Toys/MMFeature/Toys_Qwen2-VL-7B-Instruct_tv.npy
#      - Data/Toys/MMFeature/Toys_Llama-3.2-11B-Vision-Instruct_tv.npy
#      - Data/Toys/MMFeature/Toys_LLAMA8B_CLIP.npy
#      - Data/Toys/TextFeature/Toys_roberta_base_512_mean.npy
#      - Data/Toys/TextFeature/Toys_Llama_3.2_1B_Instruct_256_mean.npy
#      - Data/Toys/TextFeature/Toys_Llama_3.1_8B_Instruct_256_mean.npy
#      - Data/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy
#      - Data/Toys/TextFeature/Toys_Qwen2_VL_7B_Instruct_256_mean.npy
#      - Data/Toys/ImageFeature/Toys_convnextv2_huge.npy
#      - Data/Toys/ImageFeature/Toys_openai_clip-vit-large-patch14.npy
#      - Data/Toys/ImageFeature/Toys_swinv2_large.npy
#      - Data/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy
#      - Data/Toys/ImageFeature/Toys_Qwen2-VL-7B-Instruct_visual.npy

python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/MMFeature/Toys_Qwen2-VL-7B-Instruct_tv.npy --label1 Qwen2-VL-7B   --feat2 /home/aiscuser/ATG/Data/Toys/MMFeature/Toys_Llama-3.2-11B-Vision-Instruct_tv.npy --label2 Llama3.2-VL-11B --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/MMFeature/Toys_LLAMA8B_CLIP.npy --label1 LLaMA+CLIP  --feat2 /home/aiscuser/ATG/Data/Toys/TextFeature/Toys_roberta_base_512_mean.npy --label2 RoBERTa  --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/TextFeature/Toys_Llama_3.2_1B_Instruct_256_mean.npy --label1 Llama3.2-1B --feat2 /home/aiscuser/ATG/Data/Toys/TextFeature/Toys_Llama_3.1_8B_Instruct_256_mean.npy --label2 Llama3.1-8B  --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy --label1 Llama3.2-VL-T --feat2 /home/aiscuser/ATG/Data/Toys/TextFeature/Toys_Qwen2_VL_7B_Instruct_256_mean.npy --label2 Qwen2-VL-T  --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/ImageFeature/Toys_openai_clip-vit-large-patch14.npy --label1 CLIP --feat2 /home/aiscuser/ATG/Data/Toys/ImageFeature/Toys_swinv2_large.npy --label2 SwinV2 --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/ImageFeature/Toys_convnextv2_huge.npy --label1 ConvNeXT --feat2 /home/aiscuser/ATG/Data/Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy --label2 Llama3.2-VL-V  --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Toys/ImageFeature/Toys_Qwen2-VL-7B-Instruct_visual.npy --label1 Qwen2-VL-V --feat2 /home/aiscuser/ATG/Data/Toys/TextFeature/Toys_Qwen2_VL_7B_Instruct_256_mean.npy --label2 Qwen2-VL-T  --dataname Toys --graph_path /home/aiscuser/ATG/Data/Toys/ToysGraph.pt

# Grocery 数据集
#      - Data/GroceryS/MMFeature/GroceryS_Llama-3.2-11B-Vision-Instruct_tv.npy
#      - Data/GroceryS/MMFeature/GroceryS_Qwen2-VL-7B-Instruct_tv.npy
#      - Data/GroceryS/MMFeature/GroceryS_LLAMA8B_CLIP.npy
#      - Data/GroceryS/TextFeature/GroceryS_roberta_base_256_mean.npy
#      - Data/GroceryS/TextFeature/GroceryS_Llama_3.2_1B_Instruct_256_mean.npy
#      - Data/GroceryS/TextFeature/GroceryS_Llama_3.1_8B_Instruct_256_mean.npy
#      - Data/GroceryS/TextFeature/GroceryS_Llama_3.2_11B_Vision_Instruct_256_mean.npy
#      - Data/GroceryS/TextFeature/GroceryS_Qwen2_VL_7B_Instruct_256_mean.npy
#      - Data/GroceryS/ImageFeature/GroceryS_convnextv2_huge.npy
#      - Data/GroceryS/ImageFeature/GroceryS_openai_clip-vit-large-patch14.npy
#      - Data/GroceryS/ImageFeature/GroceryS_swinv2_large.npy
#      - Data/GroceryS/ImageFeature/GroceryS_Llama-3.2-11B-Vision-Instruct_visual.npy
#      - Data/GroceryS/ImageFeature/GroceryS_Qwen2-VL-7B-Instruct_visual.npy

python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/MMFeature/GroceryS_Qwen2-VL-7B-Instruct_tv.npy --label1 Qwen2-VL-7B   --feat2 /home/aiscuser/ATG/Data/GroceryS/MMFeature/GroceryS_Llama-3.2-11B-Vision-Instruct_tv.npy --label2 Llama3.2-VL-11B --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/MMFeature/GroceryS_LLAMA8B_CLIP.npy --label1 LLaMA+CLIP  --feat2 /home/aiscuser/ATG/Data/GroceryS/TextFeature/GroceryS_roberta_base_256_mean.npy --label2 RoBERTa  --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/TextFeature/GroceryS_Llama_3.2_1B_Instruct_256_mean.npy --label1 Llama3.2-1B --feat2 /home/aiscuser/ATG/Data/GroceryS/TextFeature/GroceryS_Llama_3.1_8B_Instruct_256_mean.npy --label2 Llama3.1-8B  --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/TextFeature/GroceryS_Llama_3.2_11B_Vision_Instruct_256_mean.npy --label1 Llama3.2-VL-T --feat2 /home/aiscuser/ATG/Data/GroceryS/TextFeature/GroceryS_Qwen2_VL_7B_Instruct_256_mean.npy --label2 Qwen2-VL-T  --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/ImageFeature/GroceryS_openai_clip-vit-large-patch14.npy --label1 CLIP --feat2 /home/aiscuser/ATG/Data/GroceryS/ImageFeature/GroceryS_swinv2_large.npy --label2 SwinV2 --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/ImageFeature/GroceryS_convnextv2_huge.npy --label1 ConvNeXT --feat2 /home/aiscuser/ATG/Data/GroceryS/ImageFeature/GroceryS_Llama-3.2-11B-Vision-Instruct_visual.npy --label2 Llama3.2-VL-V  --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/GroceryS/ImageFeature/GroceryS_Qwen2-VL-7B-Instruct_visual.npy --label1 Qwen2-VL-V --feat2 /home/aiscuser/ATG/Data/GroceryS/TextFeature/GroceryS_Qwen2_VL_7B_Instruct_256_mean.npy --label2 Qwen2-VL-T  --dataname GroceryS --graph_path /home/aiscuser/ATG/Data/GroceryS/GrocerySGraph.pt

# Reddit-S 数据集
#      - Data/RedditS/MMFeature/RedditS_Llama-3.2-11B-Vision-Instruct_tv.npy
#      - Data/RedditS/MMFeature/RedditS_Qwen2-VL-7B-Instruct_tv.npy
#      - Data/RedditS/MMFeature/RedditS_LLAMA8B_CLIP.npy
#      - Data/RedditS/TextFeature/RedditS_roberta_base_100_mean.npy
#      - Data/RedditS/TextFeature/RedditS_Llama_3.2_1B_Instruct_100_mean.npy
#      - Data/RedditS/TextFeature/RedditS_Llama_3.1_8B_Instruct_100_mean.npy
#      - Data/RedditS/TextFeature/RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy
#      - Data/RedditS/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy
#      - Data/RedditS/ImageFeature/RedditS_convnextv2.npy
#      - Data/RedditS/ImageFeature/RedditS_swinv2_large.npy
#      - Data/RedditS/ImageFeature/RedditS_openai_clip-vit-large-patch14.npy
#      - Data/RedditS/ImageFeature/RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy
#      - Data/RedditS/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy

python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/MMFeature/RedditS_Qwen2-VL-7B-Instruct_tv.npy --label1 Qwen2-VL-7B   --feat2 /home/aiscuser/ATG/Data/RedditS/MMFeature/RedditS_Llama-3.2-11B-Vision-Instruct_tv.npy --label2 Llama3.2-VL-11B --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/MMFeature/RedditS_LLAMA8B_CLIP.npy --label1 LLaMA+CLIP  --feat2 /home/aiscuser/ATG/Data/RedditS/TextFeature/RedditS_roberta_base_100_mean.npy --label2 RoBERTa  --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/TextFeature/RedditS_Llama_3.2_1B_Instruct_100_mean.npy --label1 Llama3.2-1B --feat2 /home/aiscuser/ATG/Data/RedditS/TextFeature/RedditS_Llama_3.1_8B_Instruct_100_mean.npy --label2 Llama3.1-8B  --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/TextFeature/RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy --label1 Llama3.2-VL-T --feat2 /home/aiscuser/ATG/Data/RedditS/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy --label2 Qwen2-VL-T  --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/ImageFeature/RedditS_openai_clip-vit-large-patch14.npy --label1 CLIP --feat2 /home/aiscuser/ATG/Data/RedditS/ImageFeature/RedditS_swinv2_large.npy --label2 SwinV2 --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/ImageFeature/RedditS_convnextv2.npy --label1 ConvNeXT --feat2 /home/aiscuser/ATG/Data/RedditS/ImageFeature/RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy --label2 Llama3.2-VL-V  --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/RedditS/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy --label1 Qwen2-VL-V --feat2 /home/aiscuser/ATG/Data/RedditS/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy --label2 Qwen2-VL-T  --dataname RedditS --graph_path /home/aiscuser/ATG/Data/RedditS/RedditSGraph.pt


# Reddit 数据集
#      - Data/Reddit/MMFeature/Reddit_Qwen2-VL-7B-Instruct_tv.npy
#      - Data/Reddit/MMFeature/Reddit_Llama-3.2-11B-Vision-Instruct_tv.npy
#      - Data/Reddit/MMFeature/Reddit_LLAMA8B_CLIP.npy
#      - Data/Reddit/TextFeature/Reddit_roberta_base_100_mean.npy
#      - Data/Reddit/TextFeature/Reddit_Llama_3.2_1B_Instruct_100_mean.npy
#      - Data/Reddit/TextFeature/Reddit_Llama_3.1_8B_Instruct_100_mean.npy
#      - Data/Reddit/TextFeature/Reddit_Llama_3.2_11B_Vision_Instruct_100_mean.npy
#      - Data/Reddit/TextFeature/Reddit_Qwen2_VL_7B_Instruct_100_mean.npy
#      - Data/Reddit/ImageFeature/Reddit_convnextv2_huge.npy
#      - Data/Reddit/ImageFeature/Reddit_swinv2_large.npy
#      - Data/Reddit/ImageFeature/Reddit_openai_clip-vit-large-patch14.npy
#      - Data/Reddit/ImageFeature/Reddit_Llama-3.2-11B-Vision-Instruct_visual.npy
#      - Data/Reddit/ImageFeature/Reddit_Qwen2-VL-7B-Instruct_visual.npy

python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/MMFeature/Reddit_Qwen2-VL-7B-Instruct_tv.npy --label1 Qwen2-VL-7B   --feat2 /home/aiscuser/ATG/Data/Reddit/MMFeature/Reddit_Llama-3.2-11B-Vision-Instruct_tv.npy --label2 Llama3.2-VL-11B --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/MMFeature/Reddit_LLAMA8B_CLIP.npy --label1 LLaMA+CLIP  --feat2 /home/aiscuser/ATG/Data/Reddit/TextFeature/Reddit_roberta_base_100_mean.npy --label2 RoBERTa  --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/TextFeature/Reddit_Llama_3.2_1B_Instruct_100_mean.npy --label1 Llama3.2-1B --feat2 /home/aiscuser/ATG/Data/Reddit/TextFeature/Reddit_Llama_3.1_8B_Instruct_100_mean.npy --label2 Llama3.1-8B  --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/TextFeature/Reddit_Llama_3.2_11B_Vision_Instruct_100_mean.npy --label1 Llama3.2-VL-T --feat2 /home/aiscuser/ATG/Data/Reddit/TextFeature/Reddit_Qwen2_VL_7B_Instruct_100_mean.npy --label2 Qwen2-VL-T  --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/ImageFeature/Reddit_openai_clip-vit-large-patch14.npy --label1 CLIP --feat2 /home/aiscuser/ATG/Data/Reddit/ImageFeature/Reddit_swinv2_large.npy --label2 SwinV2 --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/ImageFeature/Reddit_convnextv2_huge.npy --label1 ConvNeXT --feat2 /home/aiscuser/ATG/Data/Reddit/ImageFeature/Reddit_Llama-3.2-11B-Vision-Instruct_visual.npy --label2 Llama3.2-VL-V  --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt
python TSNE.py --sample 5000 --feat1 /home/aiscuser/ATG/Data/Reddit/ImageFeature/Reddit_Qwen2-VL-7B-Instruct_visual.npy --label1 Qwen2-VL-V --feat2 /home/aiscuser/ATG/Data/Reddit/TextFeature/Reddit_Qwen2_VL_7B_Instruct_100_mean.npy --label2 Qwen2-VL-T  --dataname Reddit --graph_path /home/aiscuser/ATG/Data/Reddit/RedditGraph.pt