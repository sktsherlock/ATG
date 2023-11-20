# 数据集处理流程

## Movies 
```shell
cd ~/ATG/Data/Movies/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz
gunzip meta_Movies_and_TV.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Movies/meta_Movies_and_TV.json' --name 'Movies' --class_numbers 20
```

## Magazine 
```shell
cd ~/ATG/Data/Magazines/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Magazine_Subscriptions.json.gz
gunzip meta_Magazine_Subscriptions.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Magazines/meta_Magazine_Subscriptions.json' --name 'Magazines' --class_numbers 17 

``` 

## Computers
```shell
cd ~/ATG/Data/Electronics/
wget --no-check-certificate https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Electronics.json.gz
gunzip meta_Electronics.json.
cd ~/ATG/Data/
python data_processing_utils_for_books.py --data_path 'Electronics/meta_Electronics.json'  --name 'Computers' --class_numbers 10 --second_category "Computers" 
```  

## Children
```shell
mkdir Data/Books/
wget --no-check-certificate https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Books.json.gz 
gunzip meta_Books.json.gz  
cd ~/ATG/Data/
python data_processing_utils_for_books.py --data_path "Books/meta_Books.json" --name "Children" --class_numbers "10" --second_category "Children"
```