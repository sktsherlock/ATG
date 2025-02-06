## GroceryS Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'answerdotai/ModernBERT-base' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 512 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'FacebookAI/roberta-base' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 512 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 100 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True

CUDA_VISIBLE_DEVICES=0,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 20 --text_column 'text' --fp16 True   #  'meta-llama/Llama-3.2-11B-Vision-Instruct'  'google/paligemma2-3b-pt-224'  'Qwen/Qwen2-VL-7B-Instruct' 
CUDA_VISIBLE_DEVICES=0,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'google/paligemma2-3b-pt-448' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True   #  'meta-llama/Llama-3.2-11B-Vision-Instruct'  'google/paligemma2-3b-pt-224'  'Qwen/Qwen2-VL-7B-Instruct'
CUDA_VISIBLE_DEVICES=3 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'Qwen/Qwen2-VL-7B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True 
```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="GroceryS_ModernBERT_base_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_ModernBERT_base_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_roberta_base_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_roberta_base_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.2_1B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.2_1B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.2_3B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.2_3B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.1_8B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.1_8B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Ministral_8B_Instruct_2410_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Ministral_8B_Instruct_2410_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.2_11B_Vision_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.2_11B_Vision_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_paligemma2_3b_pt_448_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_paligemma2_3b_pt_448_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Qwen2_VL_7B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Qwen2_VL_7B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
```


## Movies Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'answerdotai/ModernBERT-base' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 400 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'FacebookAI/roberta-base' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 400 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 50 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 30 --text_column 'text' --f16 True --fp16 True   # 12/16G 
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 5 --text_column 'text' --f16 True --fp16 True  
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 40 --text_column 'text' --f16 True --fp16 True

CUDA_VISIBLE_DEVICES=0,2 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 20 --f16 True --text_column 'text' --fp16 True  #32G
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'google/paligemma2-3b-pt-448' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 50 --f16 True --text_column 'text' --fp16 True  #32G 
CUDA_VISIBLE_DEVICES=3 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'Qwen/Qwen2-VL-7B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 50 --f16 True --text_column 'text' --fp16 True  #32G
```



```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="Movies_Llama_3.1_8B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.1_8B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Llama_3.2_1B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.2_1B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Llama_3.2_3B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.2_3B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_ModernBERT_base_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_ModernBERT_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_roberta_base_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_roberta_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Ministral_8B_Instruct_2410_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Ministral_8B_Instruct_2410_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_paligemma2_3b_pt_448_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_paligemma2_3b_pt_448_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Qwen2_VL_7B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Qwen2_VL_7B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
```


## Toys Text Feature 
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'answerdotai/ModernBERT-base' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'FacebookAI/roberta-base' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 200 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2,5,6 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True

CUDA_VISIBLE_DEVICES=0,2 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 30 --text_column 'text' --fp16 True 
CUDA_VISIBLE_DEVICES=3 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'google/paligemma2-3b-pt-448' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=3 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'Qwen/Qwen2-VL-7B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 40 --text_column 'text' --fp16 True


```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="Toys_ModernBERT_base_512_mean.npy", path_in_repo="Toys/TextFeature/Toys_ModernBERT_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_roberta_base_512_mean.npy", path_in_repo="Toys/TextFeature/Toys_roberta_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.2_1B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.2_1B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.2_3B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.2_3B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.1_8B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.1_8B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Ministral_8B_Instruct_2410_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Ministral_8B_Instruct_2410_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.2_11B_Vision_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_paligemma2_3b_pt_448_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_paligemma2_3b_pt_448_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Qwen2_VL_7B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Qwen2_VL_7B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
```

## RedditS Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'answerdotai/ModernBERT-base' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 500 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'FacebookAI/roberta-base' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 500 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 100 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 50 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True

CUDA_VISIBLE_DEVICES=0,2 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 20 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=2 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'google/paligemma2-3b-pt-448' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 50 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=3 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'Qwen/Qwen2-VL-7B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True

```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="RedditS_ModernBERT_base_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_ModernBERT_base_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_roberta_base_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_roberta_base_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.2_1B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.2_1B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.2_3B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.2_3B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.1_8B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.1_8B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Ministral_8B_Instruct_2410_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Ministral_8B_Instruct_2410_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.2_11B_Vision_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_paligemma2_3b_pt_448_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_paligemma2_3b_pt_448_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Qwen2_VL_7B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Qwen2_VL_7B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
```

## Reddit Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'answerdotai/ModernBERT-base' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 500 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'FacebookAI/roberta-base' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 500 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 100 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 50 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True

CUDA_VISIBLE_DEVICES=0,2 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 20 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=2 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'google/paligemma2-3b-pt-448' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 100 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=3 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'Qwen/Qwen2-VL-7B-Instruct' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True
```

```

upload_file(path_or_fileobj="Reddit_ModernBERT_base_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_ModernBERT_base_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_roberta_base_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_roberta_base_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Llama_3.2_1B_Instruct_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_Llama_3.2_1B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Llama_3.2_3B_Instruct_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_Llama_3.2_3B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Llama_3.1_8B_Instruct_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_Llama_3.1_8B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Ministral_8B_Instruct_2410_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_Ministral_8B_Instruct_2410_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Llama_3.2_11B_Vision_Instruct_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_Llama_3.2_11B_Vision_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_paligemma2_3b_pt_448_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_paligemma2_3b_pt_448_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Qwen2_VL_7B_Instruct_100_mean.npy", path_in_repo="Reddit/TextFeature/Reddit_Qwen2_VL_7B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
```

## MLLM TV Feature
```python
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/MMFeature/  --name Movies --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' 
CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/MMFeature/  --name Toys  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct'
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/MMFeature/  --name RedditS  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct'
CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/ImageFeature/  --name RedditS  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --feature_type 'visual'
```

```python
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/MMFeature/  --name GroceryS  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' 
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/ImageFeature/  --name GroceryS  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --feature_type 'visual'
```

CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/MMFeature/  --name Movies --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' 
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/ImageFeature/  --name Movies --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --feature_type 'visual'

CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/MMFeature/  --name Toys  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct'
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/ImageFeature/  --name Toys  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --feature_type 'visual'


```python
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/MMFeature/  --name Reddit  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct'
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/ImageFeature/  --name Reddit  --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct' --feature_type 'visual'

```

```python
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/ImageFeature/  --name Movies --model_name 'meta-llama/Llama-3.2-11B-Vision-Instruct'  --feature_type 'visual'
```


```python
#Qwen2B
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/MMFeature/  --name Movies --model_name 'Qwen/Qwen2-VL-2B-Instruct' 
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/MMFeature/    --name GroceryS --model_name  'Qwen/Qwen2-VL-2B-Instruct'
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/MMFeature/  --name RedditS  --model_name  'Qwen/Qwen2-VL-2B-Instruct'
CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/MMFeature/  --name Toys --model_name  'Qwen/Qwen2-VL-2B-Instruct'
CUDA_VISIBLE_DEVICES=4 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/MMFeature/  --name Reddit  --model_name  'Qwen/Qwen2-VL-2B-Instruct'  --text_column 'caption'
#Qwen7B
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/MMFeature/  --name Movies --model_name 'Qwen/Qwen2-VL-7B-Instruct' 
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/ImageFeature/  --name Movies --model_name 'Qwen/Qwen2-VL-7B-Instruct'  --feature_type 'visual'
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/MMFeature/    --name GroceryS --model_name  'Qwen/Qwen2-VL-7B-Instruct' 
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/ImageFeature/    --name GroceryS --model_name  'Qwen/Qwen2-VL-7B-Instruct'  --feature_type 'visual'
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/MMFeature/  --name RedditS  --model_name  'Qwen/Qwen2-VL-7B-Instruct'  --text_column 'caption'
CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/ImageFeature/  --name RedditS  --model_name  'Qwen/Qwen2-VL-7B-Instruct'  --text_column 'caption'  --feature_type 'visual'
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/MMFeature/  --name Toys --model_name  'Qwen/Qwen2-VL-7B-Instruct'
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/ImageFeature/  --name Toys --model_name  'Qwen/Qwen2-VL-7B-Instruct'  --feature_type 'visual'
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/MMFeature/  --name Reddit  --model_name  'Qwen/Qwen2-VL-7B-Instruct'  --text_column 'caption'
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/ImageFeature/  --name Reddit  --model_name  'Qwen/Qwen2-VL-7B-Instruct'  --text_column 'caption'  --feature_type 'visual'
```

```python
#PaliGemma
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/MMFeature/  --name Movies --model_name 'google/paligemma2-3b-pt-448'
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/MMFeature/    --name GroceryS --model_name  'google/paligemma2-3b-pt-448'
CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/MMFeature/  --name RedditS  --model_name  'google/paligemma2-3b-pt-224'  --text_column 'caption'
CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/MMFeature/  --name Toys --model_name  'google/paligemma2-3b-pt-448'
CUDA_VISIBLE_DEVICES=4 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/MMFeature/  --name Reddit  --model_name  'google/paligemma2-3b-pt-896'  --text_column 'caption'
```

```python
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/MMFeature/ --name RedditS  --model_name  'google/paligemma2-3b-pt-448'  --text_column 'caption'  --feature_type 'tv'
CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/RedditS/RedditS.csv --image_path /home/aiscuser/ATG/Data/RedditS/RedditSImages/ --path /home/aiscuser/ATG/Data/RedditS/ImageFeature/ --name RedditS  --model_name  'google/paligemma2-3b-pt-448'  --text_column 'caption'  --feature_type 'visual'


CUDA_VISIBLE_DEVICES=3 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/MMFeature/  --name Toys --model_name  'google/paligemma2-3b-pt-448'  --feature_type 'tv'
CUDA_VISIBLE_DEVICES=4 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --image_path /home/aiscuser/ATG/Data/Toys/ToysImages/ --path /home/aiscuser/ATG/Data/Toys/ImageFeature/  --name Toys --model_name  'google/paligemma2-3b-pt-448'  --feature_type 'visual'

# GroceryS
CUDA_VISIBLE_DEVICES=5 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/MMFeature/  --name GroceryS --model_name 'google/paligemma2-3b-pt-448'  --feature_type 'tv'
CUDA_VISIBLE_DEVICES=6 python MLLM.py --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --image_path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --path /home/aiscuser/ATG/Data/GroceryS/ImageFeature/  --name GroceryS --model_name 'google/paligemma2-3b-pt-448'  --feature_type 'visual'

CUDA_VISIBLE_DEVICES=0 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/MMFeature/  --name Reddit  --model_name  'google/paligemma2-3b-pt-448'  --text_column 'caption'  --feature_type 'tv'
CUDA_VISIBLE_DEVICES=7 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --image_path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --path /home/aiscuser/ATG/Data/Reddit/ImageFeature/  --name Reddit  --model_name  'google/paligemma2-3b-pt-448'  --text_column 'caption'  --feature_type 'visual'

# Movies
CUDA_VISIBLE_DEVICES=1 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/MMFeature/  --name Movies --model_name 'google/paligemma2-3b-pt-448'  --feature_type 'tv'
CUDA_VISIBLE_DEVICES=2 python MLLM.py --csv_path /home/aiscuser/ATG/Data/Movies/Movies.csv --image_path /home/aiscuser/ATG/Data/Movies/MoviesImages/ --path /home/aiscuser/ATG/Data/Movies/ImageFeature/  --name Movies --model_name 'google/paligemma2-3b-pt-448'  --feature_type 'visual'



from huggingface_hub import upload_file
upload_file(path_or_fileobj="RedditS_Qwen2-VL-2B-Instruct_tv.npy", path_in_repo="RedditS/MMFeature/RedditS_Qwen2-VL-2B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Qwen2-VL-2B-Instruct_tv.npy", path_in_repo="Movies/MMFeature/Movies_Qwen2-VL-2B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Qwen2-VL-2B-Instruct_tv.npy", path_in_repo="Toys/MMFeature/Toys_Qwen2-VL-2B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Qwen2-VL-2B-Instruct_tv.npy", path_in_repo="GroceryS/MMFeature/GroceryS_Qwen2-VL-2B-Instruct_tv.npy", repo_id="Sherirto/MAG")

from huggingface_hub import upload_file
upload_file(path_or_fileobj="RedditS_Llama-3.2-11B-Vision-Instruct_tv.npy", path_in_repo="RedditS/MMFeature/RedditS_Llama-3.2-11B-Vision-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama-3.2-11B-Vision-Instruct_tv.npy", path_in_repo="GroceryS/MMFeature/GroceryS_Llama-3.2-11B-Vision-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Llama-3.2-11B-Vision-Instruct_tv.npy", path_in_repo="Movies/MMFeature/Movies_Llama-3.2-11B-Vision-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama-3.2-11B-Vision-Instruct_tv.npy", path_in_repo="Toys/MMFeature/Toys_Llama-3.2-11B-Vision-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Llama-3.2-11B-Vision-Instruct_tv.npy", path_in_repo="Reddit/MMFeature/Reddit_Llama-3.2-11B-Vision-Instruct_tv.npy", repo_id="Sherirto/MAG")


upload_file(path_or_fileobj="Movies_Qwen2-VL-7B-Instruct_tv.npy", path_in_repo="Movies/MMFeature/Movies_Qwen2-VL-7B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Qwen2-VL-7B-Instruct_tv.npy", path_in_repo="GroceryS/MMFeature/GroceryS_Qwen2-VL-7B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Qwen2-VL-7B-Instruct_tv.npy", path_in_repo="Toys/MMFeature/Toys_Qwen2-VL-7B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Qwen2-VL-7B-Instruct_tv.npy", path_in_repo="RedditS/MMFeature/RedditS_Qwen2-VL-7B-Instruct_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Qwen2-VL-7B-Instruct_tv.npy", path_in_repo="Reddit/MMFeature/Reddit_Qwen2-VL-7B-Instruct_tv.npy", repo_id="Sherirto/MAG")


upload_file(path_or_fileobj="RedditS_paligemma2-3b-pt-448_tv.npy", path_in_repo="RedditS/MMFeature/RedditS_paligemma2-3b-pt-448_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_paligemma2-3b-pt-448_visual.npy", path_in_repo="RedditS/ImageFeature/RedditS_paligemma2-3b-pt-448_visual.npy", repo_id="Sherirto/MAG")

upload_file(path_or_fileobj="Reddit_paligemma2-3b-pt-448_tv.npy", path_in_repo="Reddit/MMFeature/Reddit_paligemma2-3b-pt-448_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_paligemma2-3b-pt-448_visual.npy", path_in_repo="Reddit/ImageFeature/Reddit_paligemma2-3b-pt-448_visual.npy", repo_id="Sherirto/MAG")

upload_file(path_or_fileobj="Movies_paligemma2-3b-pt-448_tv.npy", path_in_repo="Movies/MMFeature/Movies_paligemma2-3b-pt-448_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_paligemma2-3b-pt-448_visual.npy", path_in_repo="Movies/ImageFeature/Movies_paligemma2-3b-pt-448_visual.npy", repo_id="Sherirto/MAG")

from huggingface_hub import upload_file
upload_file(path_or_fileobj="Toys_paligemma2-3b-pt-448_tv.npy", path_in_repo="Toys/MMFeature/Toys_paligemma2-3b-pt-448_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_paligemma2-3b-pt-448_visual.npy", path_in_repo="Toys/ImageFeature/Toys_paligemma2-3b-pt-448_visual.npy", repo_id="Sherirto/MAG")

from huggingface_hub import upload_file
upload_file(path_or_fileobj="GroceryS_paligemma2-3b-pt-448_tv.npy", path_in_repo="GroceryS/MMFeature/GroceryS_paligemma2-3b-pt-448_tv.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_paligemma2-3b-pt-448_visual.npy", path_in_repo="GroceryS/ImageFeature/GroceryS_paligemma2-3b-pt-448_visual.npy", repo_id="Sherirto/MAG")

from huggingface_hub import upload_file
upload_file(path_or_fileobj="Movies_Qwen2-VL-7B-Instruct_visual.npy", path_in_repo="Movies/ImageFeature/Movies_Qwen2-VL-7B-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Qwen2-VL-7B-Instruct_visual.npy", path_in_repo="Toys/ImageFeature/Toys_Qwen2-VL-7B-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Qwen2-VL-7B-Instruct_visual.npy", path_in_repo="GroceryS/ImageFeature/GroceryS_Qwen2-VL-7B-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Qwen2-VL-7B-Instruct_visual.npy", path_in_repo="RedditS/ImageFeature/RedditS_Qwen2-VL-7B-Instruct_visual.npy", repo_id="Sherirto/MAG")

upload_file(path_or_fileobj="Movies_Llama-3.2-11B-Vision-Instruct_visual.npy", path_in_repo="Movies/ImageFeature/Movies_Llama-3.2-11B-Vision-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama-3.2-11B-Vision-Instruct_visual.npy", path_in_repo="Toys/ImageFeature/Toys_Llama-3.2-11B-Vision-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama-3.2-11B-Vision-Instruct_visual.npy", path_in_repo="GroceryS/ImageFeature/GroceryS_Llama-3.2-11B-Vision-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Reddit_Llama-3.2-11B-Vision-Instruct_visual.npy", path_in_repo="Reddit/ImageFeature/Reddit_Llama-3.2-11B-Vision-Instruct_visual.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy", path_in_repo="RedditS/ImageFeature/RedditS_Llama-3.2-11B-Vision-Instruct_visual.npy", repo_id="Sherirto/MAG")
```

```python
upload_file(path_or_fileobj="Reddit_LLAMA8B_CLIP.npy", path_in_repo="Reddit/MMFeature/Reddit_LLAMA8B_CLIP.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_LLAMA8B_CLIP.npy", path_in_repo="RedditS/MMFeature/RedditS_LLAMA8B_CLIP.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_LLAMA8B_CLIP.npy", path_in_repo="Movies/MMFeature/Movies_LLAMA8B_CLIP.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_LLAMA8B_CLIP.npy", path_in_repo="Toys/MMFeature/Toys_LLAMA8B_CLIP.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_LLAMA8B_CLIP.npy", path_in_repo="GroceryS/MMFeature/GroceryS_LLAMA8B_CLIP.npy", repo_id="Sherirto/MAG")
```
