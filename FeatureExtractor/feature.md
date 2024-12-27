## GroceryS Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'answerdotai/ModernBERT-base' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 512 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'FacebookAI/roberta-base' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 512 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 100 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="GroceryS_ModernBERT_base_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_ModernBERT_base_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_roberta_base_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_roberta_base_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.2_1B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.2_1B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.2_3B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.2_3B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Llama_3.1_8B_Instruct_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Llama_3.1_8B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="GroceryS_Ministral_8B_Instruct_2410_256_mean.npy", path_in_repo="GroceryS/TextFeature/GroceryS_Ministral_8B_Instruct_2410_256_mean.npy", repo_id="Sherirto/MAG")
```


## Movies Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'answerdotai/ModernBERT-base' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 400 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'FacebookAI/roberta-base' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 400 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 50 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 30 --text_column 'text' --f16 True --fp16 True   # 12/16G 
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 5 --text_column 'text' --f16 True --fp16 True  
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 40 --text_column 'text' --f16 True --fp16 True
```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="Movies_Llama_3.1_8B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.1_8B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Llama_3.2_1B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.2_1B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_Llama_3.2_3B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.2_3B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_ModernBERT_base_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_ModernBERT_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Movies_roberta_base_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_roberta_base_512_mean.npy", repo_id="Sherirto/MAG")
```


## Toys Text Feature 
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'answerdotai/ModernBERT-base' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'FacebookAI/roberta-base' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 200 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2,5,6 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="Toys_ModernBERT_base_512_mean.npy", path_in_repo="Toys/TextFeature/Toys_ModernBERT_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_roberta_base_512_mean.npy", path_in_repo="Toys/TextFeature/Toys_roberta_base_512_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.2_1B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.2_1B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.2_3B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.2_3B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Llama_3.1_8B_Instruct_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Llama_3.1_8B_Instruct_256_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="Toys_Ministral_8B_Instruct_2410_256_mean.npy", path_in_repo="Toys/TextFeature/Toys_Ministral_8B_Instruct_2410_256_mean.npy", repo_id="Sherirto/MAG")
```

## RedditS Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'answerdotai/ModernBERT-base' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 500 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'FacebookAI/roberta-base' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 500 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 100 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 50 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/RedditS/RedditS.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'RedditS' --path 'Data/RedditS/TextFeature/' --max_length 100 --batch_size 40 --text_column 'caption' --fp16 True
```

```python
from huggingface_hub import upload_file
upload_file(path_or_fileobj="RedditS_ModernBERT_base_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_ModernBERT_base_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_roberta_base_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_roberta_base_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.2_1B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.2_1B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.2_3B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.2_3B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Llama_3.1_8B_Instruct_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Llama_3.1_8B_Instruct_100_mean.npy", repo_id="Sherirto/MAG")
upload_file(path_or_fileobj="RedditS_Ministral_8B_Instruct_2410_100_mean.npy", path_in_repo="RedditS/TextFeature/RedditS_Ministral_8B_Instruct_2410_100_mean.npy", repo_id="Sherirto/MAG")
```