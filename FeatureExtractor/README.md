# Text Attribute -> Feature 

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