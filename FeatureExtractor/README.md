# Text Attribute -> Feature 

# MAG 
## Movies Text Feature
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 20 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 20 --text_column 'text' --f16 True --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --norm True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Movies/Movies.csv --model_name 'google/gemma-7b' --name 'Movies' --path 'Data/Movies/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' 
```

## Photo Text Feature
```python
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Photo/Photo.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'Photo' --path 'Data/Photo/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Photo/Photo.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'Photo' --path 'Data/Photo/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Photo/Photo.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'Photo' --path 'Data/Photo/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --norm True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Photo/Photo.csv --model_name 'google/gemma-7b' --name 'Photo' --path 'Data/Photo/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' 
```

## Photo Image Feature
```python
python ImageExtract.py --gpu 0 --data_dir Data/Photo/PhotoImages/ --name Movies --path Data/Photo/ImageFeature/ --batch_size 64 --model_name vit_large_patch14_dinov2.lvd142m --size 224
python ImageExtract.py --gpu 0 --data_dir Data/Photo/PhotoImages/ --name Movies --path Data/Photo/ImageFeature/ --batch_size 64 --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k --size 384
python ImageExtract.py --gpu 1 --data_dir Data/Photo/PhotoImages/ --name Movies --path Data/Photo/ImageFeature/ --batch_size 1024 --model_name convnextv2_huge.fcmae_ft_in22k_in1k_384
```


## Grocery Text Feature
```python
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Grocery/Grocery.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'Grocery' --path 'Data/Grocery/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Grocery/Grocery.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'Grocery' --path 'Data/Grocery/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Grocery/Grocery.csv --model_name 'google/gemma-7b' --name 'Grocery' --path 'Data/Grocery/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Grocery/Grocery.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'Grocery' --path 'Data/Grocery/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --norm True
```
### Grocery Image Feature
```python
CUDA_VISIBLE_DEVICES=0 python CLIP.py --name Grocery --csv_path /home/aiscuser/ATG/Data/Grocery/Grocery.csv --path /home/aiscuser/ATG/Data/Grocery/GroceryImages/ --feature_path /home/aiscuser/ATG/Data/Grocery/ImageFeature
python ImageExtract.py --gpu 3 --data_dir Data/Grocery/GroceryImages/ --name Grocery --path Data/Grocery/ImageFeature/ --batch_size 64 --model_name vit_large_patch14_dinov2.lvd142m --size 518
python ImageExtract.py --gpu 4 --data_dir Data/Grocery/GroceryImages/ --name Grocery --path Data/Grocery/ImageFeature/ --batch_size 64 --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k --size 384
python ImageExtract.py --gpu 5 --data_dir Data/Grocery/GroceryImages/ --name Grocery --path Data/Grocery/ImageFeature/ --batch_size 1024 --model_name convnextv2_huge.fcmae_ft_in22k_in1k_384
```

## GroceryS Text Feature
```python
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'google/gemma-7b' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/GroceryS/GroceryS.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'GroceryS' --path 'Data/GroceryS/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --norm True
```
### GroceryS Image Feature
```python
CUDA_VISIBLE_DEVICES=0 python CLIP.py --name GroceryS --csv_path /home/aiscuser/ATG/Data/GroceryS/GroceryS.csv --path /home/aiscuser/ATG/Data/GroceryS/GrocerySImages/ --feature_path /home/aiscuser/ATG/Data/GroceryS/ImageFeature
python ImageExtract.py  --data_dir Data/GroceryS/GrocerySImages/ --name GroceryS --path Data/GroceryS/ImageFeature/ --batch_size 64 --model_name vit_large_patch14_dinov2.lvd142m --size 518
python ImageExtract.py  --data_dir Data/GroceryS/GrocerySImages/ --name GroceryS --path Data/GroceryS/ImageFeature/ --batch_size 64 --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k --size 384
python ImageExtract.py  --data_dir Data/GroceryS/GrocerySImages/ --name GroceryS --path Data/GroceryS/ImageFeature/ --batch_size 1024 --model_name convnextv2_huge.fcmae_ft_in22k_in1k_384
```

## Toys Text Feature 
```python
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'google/gemma-7b' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True 
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Toys/Toys.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'Toys' --path 'Data/Toys/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --fp16 True --norm True
```
CUDA_VISIBLE_DEVICES=0 python CLIP.py --name Toys --csv_path /home/aiscuser/ATG/Data/Toys/Toys.csv --path /home/aiscuser/ATG/Data/Toys/ToysImages/ --feature_path /home/aiscuser/ATG/Data/Toys/ImageFeature
CUDA_VISIBLE_DEVICES=0  python ImageExtract.py  --data_dir Data/Toys/ToysImages/ --name Toys --path Data/Toys/ImageFeature/ --batch_size 64 --model_name vit_large_patch14_dinov2.lvd142m --size 518
python ImageExtract.py  --data_dir Data/Toys/ToysImages/ --name Toys --path Data/Toys/ImageFeature/ --batch_size 64 --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k --size 384
python ImageExtract.py  --data_dir Data/Toys/ToysImages/ --name Toys --path Data/Toys/ImageFeature/ --batch_size 512 --model_name convnextv2_huge.fcmae_ft_in22k_in1k_384



## Reddit Text Feature
```python
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 80 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 80 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'google/gemma-7b' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 80 --text_column 'text' --fp16 True 
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file Data/Reddit/Reddit.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'Reddit' --path 'Data/Reddit/TextFeature/' --max_length 100 --batch_size 500 --text_column 'text' --fp16 True --norm True
```
CUDA_VISIBLE_DEVICES=0 python CLIP.py --name Reddit --csv_path /home/aiscuser/ATG/Data/Reddit/Reddit.csv --path /home/aiscuser/ATG/Data/Reddit/RedditImages/ --feature_path /home/aiscuser/ATG/Data/Reddit/ImageFeature
CUDA_VISIBLE_DEVICES=0  python ImageExtract.py  --data_dir Data/Reddit/RedditImages/ --name Reddit --path Data/Reddit/ImageFeature/ --batch_size 64 --model_name vit_large_patch14_dinov2.lvd142m --size 518
python ImageExtract.py  --data_dir Data/Reddit/RedditImages/ --name Reddit --path Data/Reddit/ImageFeature/ --batch_size 128 --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k --size 384
python ImageExtract.py  --data_dir Data/Reddit/RedditImages/ --name Reddit --path Data/Reddit/ImageFeature/ --batch_size 512 --model_name convnextv2_huge.fcmae_ft_in22k_in1k_384


## Arts Text Feature
```python
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Arts/Arts.csv --model_name 'meta-llama/Llama-2-7b-hf' --name 'Arts' --path 'Data/Arts/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text' --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Arts/Arts.csv --model_name 'mistralai/Mistral-7B-v0.1' --name 'Arts' --path 'Data/Arts/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file Data/Arts/Arts.csv --model_name 'google/gemma-7b' --name 'Arts' --path 'Data/Arts/TextFeature/' --max_length 256 --batch_size 50 --text_column 'text'  --fp16 True
CUDA_VISIBLE_DEVICES=1 python TextAttributeExtract.py --csv_file Data/Arts/Arts.csv --model_name 'sentence-transformers/all-MiniLM-L12-v2' --name 'Arts' --path 'Data/Arts/TextFeature/' --max_length 512 --batch_size 500 --text_column 'text' --norm True
```
### Arts Image Feature
```python
CUDA_VISIBLE_DEVICES=1 python CLIP.py --name Arts --csv_path /home/aiscuser/ATG/Data/Arts/Arts.csv --path /home/aiscuser/ATG/Data/Arts/ArtsImages/ --feature_path /home/aiscuser/ATG/Data/Arts/ImageFeature
CUDA_VISIBLE_DEVICES=4 python ImageExtract.py  --data_dir Data/Arts/ArtsImages/ --name Arts --path Data/Arts/ImageFeature/ --batch_size 64 --model_name vit_large_patch14_dinov2.lvd142m --size 518
CUDA_VISIBLE_DEVICES=5 python ImageExtract.py --data_dir Data/Arts/ArtsImages/ --name Arts --path Data/Arts/ImageFeature/ --batch_size 128 --model_name swinv2_large_window12to24_192to384.ms_in22k_ft_in1k --size 384
CUDA_VISIBLE_DEVICES=6 python ImageExtract.py --data_dir Data/Arts/ArtsImages/ --name Arts --path Data/Arts/ImageFeature/ --batch_size 256 --model_name convnextv2_huge.fcmae_ft_in22k_in1k_384
```

### Reddit Image Feature




## Movies 
```python
# BERT-related
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'prajjwal1/bert-tiny' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'prajjwal1/bert-mini' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'bert-base-uncased' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'bert-large-uncased' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 

# RoBERTa-related 
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'roberta-base' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'roberta-large' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 

# DistilBERT-related 
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'distilbert-base-uncased' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 

# DistilRoBERTa-related
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'distilroberta-base' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 1000 --cls 

# OPT
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'facebook/opt-1.3b' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 500 

# MPT
python TextAttributeExtract.py --csv_file 'Data/Movies/Movies.csv' --model_name 'mosaicml/mpt-7b' --name 'Movies' --path 'Data/Movies/Feature/' --max_length 128 --batch_size 500

```

## Arxiv
```python
python TextAttributeExtract.py --csv_file 'Data/ogb/arxiv.csv' --model_name 'bert-base-uncased' --name 'Arxiv' --path 'Data/ogb/Arxiv/Feature/' --max_length 256 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file 'Data/ogb/arxiv.csv' --model_name 'bert-large-uncased' --name 'Arxiv' --path 'Data/ogb/Arxiv/Feature/' --max_length 256 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Arxiv.csv' --model_name 'roberta-large' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 256 --batch_size 1000 --cls 
python TextAttributeExtract.py --csv_file 'Data/ogb/arxiv.csv' --model_name 'roberta-base' --name 'Arxiv' --path 'Data/ogb/Arxiv/Feature/' --max_length 256 --batch_size 1000 --cls 

python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/arxiv.csv' --model_name 'bert-large-uncased' --name 'Arxiv' --path 'Data/ogb/Arxiv/Category/' --max_length 128 --batch_size 2000 --cls --text_column 'TC'


# OPT 32GB V100
python TextAttributeExtract.py --csv_file 'Data/ogb/arxiv.csv' --model_name 'facebook/opt-1.3b' --name 'Arxiv' --path 'Data/ogb/Arxiv/Feature/' --max_length 256 --batch_size 200 
python TextAttributeExtract.py --csv_file 'Data/ogb/arxiv.csv' --model_name 'facebook/opt-1.3b' --name 'Arxiv' --path 'Data/ogb/Arxiv/Feature/' --max_length 256 --batch_size 200 


```


```python
CUDA_VISIBLE_DEVICES=0 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/OGBN_ARXIV.csv' --model_name 'bert-large-uncased' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 512 --batch_size 500 --cls --text_column 'TA'



# LLM

CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/OGBN_ARXIV.csv' --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 512 --batch_size 200 --text_column 'TA'   [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Movies.csv' --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --name 'Movies' --path '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Feature/' --max_length 512 --batch_size 200 --text_column 'text'           [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Children.csv' --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --name 'Children' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/' --max_length 512 --batch_size 200 --text_column 'text'  [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/History.csv' --model_name 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' --name 'History' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/' --max_length 512 --batch_size 200 --text_column 'text'   [OK]

CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/OGBN_ARXIV.csv' --model_name 'meta-llama/Llama-2-7b-hf' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 256 --batch_size 50 --text_column 'TA'             [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Movies.csv' --model_name 'meta-llama/Llama-2-7b-hf' --name 'Movies' --path '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Feature/' --max_length 256 --batch_size 50 --text_column 'text'   [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Children.csv' --model_name 'meta-llama/Llama-2-7b-hf' --name 'Children' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/' --max_length 256 --batch_size 50 --text_column 'text'  [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/History.csv' --model_name 'meta-llama/Llama-2-7b-hf' --name 'History' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/' --max_length 256 --batch_size 50 --text_column 'text' 



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/OGBN_ARXIV.csv' --model_name 'meta-llama/Llama-2-13b-hf' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 256 --batch_size 10 --text_column 'TA'     [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Movies.csv' --model_name 'meta-llama/Llama-2-13b-hf' --name 'Movies' --path '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Feature/' --max_length 256 --batch_size 10 --text_column 'text'  [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Children.csv' --model_name 'meta-llama/Llama-2-13b-hf' --name 'Children' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/' --max_length 256 --batch_size 10 --text_column 'text' [OK]  
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/History.csv' --model_name 'meta-llama/Llama-2-13b-hf' --name 'History' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/' --max_length 256 --batch_size 10 --text_column 'text'   [OK]  


Milpt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/OGBN_ARXIV.csv' --model_name 'mistralai/Mistral-7B-v0.1' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 256 --batch_size 50 --text_column 'TA'             [OK]
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Movies.csv' --model_name 'mistralai/Mistral-7B-v0.1' --name 'Movies' --path '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Feature/' --max_length 256 --batch_size 50 --text_column 'text'  

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Children.csv' --model_name 'mistralai/Mistral-7B-v0.1' --name 'Children' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/' --max_length 256 --batch_size 50 --text_column 'text' 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/History.csv' --model_name 'mistralai/Mistral-7B-v0.1' --name 'History' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/' --max_length 256 --batch_size 50 --text_column 'text' 



T-PLM 
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Children.csv' --model_name 'roberta-base' --pretrain_path '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/TPLM/RoBERTa' --name 'Children' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/' --max_length 512 --batch_size 500 --text_column 'text' --cls
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/History.csv' --model_name 'roberta-base' --pretrain_path '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/TPLM/RoBERTa' --name 'History' --path '/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/' --max_length 512 --batch_size 500 --text_column 'text' --cls
CUDA_VISIBLE_DEVICES=0,1 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Movies.csv' --model_name 'roberta-base' --pretrain_path '/dataintent/local/user/v-yinju/haoyan/Data/Movies/TPLM/RoBERTa' --name 'Movies' --path '/dataintent/local/user/v-yinju/haoyan/Data/Movies/Feature/' --max_length 512 --batch_size 500 --text_column 'text' --cls
CUDA_VISIBLE_DEVICES=0,1,2,3 python TextAttributeExtract.py --csv_file '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/OGBN_ARXIV.csv' --model_name 'roberta-base' --pretrain_path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/TPLM/RoBERTa' --name 'Arxiv' --path '/dataintent/local/user/v-yinju/haoyan/Data/OGB/Arxiv/Feature/' --max_length 512 --batch_size 500 --text_column 'TA' --cls



```

# Image Attribute -> Feature 
## Movies
```python 
# ResNet-50 
python ImageExtract.py --data_dir 'Data/Movies/MoviesImages/' --model_name 'resnet50d' --name 'Movies' --path 'Data/Movies/ImageFeature/' --batch_size 200 --pretrained True --size 224
# ResNet-101 
python ImageExtract.py --data_dir 'Data/Movies/MoviesImages/' --model_name 'resnet101d' --name 'Movies' --path 'Data/Movies/ImageFeature/' --batch_size 200 --pretrained True --size 224
# ResNet-152 
python ImageExtract.py --data_dir 'Data/Movies/MoviesImages/' --model_name 'resnet152d' --name 'Movies' --path 'Data/Movies/ImageFeature/' --batch_size 200 --pretrained True --size 224 
# DenseNet-121

```