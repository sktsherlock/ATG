## GroceryS Text Feature
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'answerdotai/ModernBERT-base' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 512 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'FacebookAI/roberta-base' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 512 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 100 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True

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
upload_file(path_or_fileobj="Movies_Llama_3.1_8B_Instruct_512_mean.npy", path_in_repo="Movies/TextFeature/Movies_Llama_3.1_8B_Instruct_512_mean.npy", repo_id="Sherirto/MAG")
```


## Toys Text Feature 
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'answerdotai/ModernBERT-base' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1,2 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'FacebookAI/roberta-base' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-1B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 400 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=3,4 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.2-3B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 400 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-3.1-8B-Instruct' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'mistralai/Ministral-8B-Instruct-2410' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
```

