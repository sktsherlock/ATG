# 数据集处理流程

## Movies 
```shell
mkdir ~/ATG/Data/Movies/
cd ~/ATG/Data/Movies/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Movies_and_TV.json.gz
gunzip meta_Movies_and_TV.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Movies/meta_Movies_and_TV.json' --name 'Movies' --class_numbers 20
```

## Magazine 
```shell
mkdir ~/ATG/Data/Magazines/
cd ~/ATG/Data/Magazines/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Magazine_Subscriptions.json.gz
gunzip meta_Magazine_Subscriptions.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Magazines/meta_Magazine_Subscriptions.json' --name 'Magazines' --class_numbers 17 

``` 

## Computers
```shell
mkdir ~/ATG/Data/Electronics/
cd ~/ATG/Data/Electronics/
wget --no-check-certificate https://jmcauley.ucsd.edu/data/amazon_v2/metaFiles2/meta_Electronics.json.gz
gunzip meta_Electronics.json.gz 
cd ~/ATG/Data/
python data_processing_utils_for_books.py --data_path 'Electronics/meta_Electronics.json'  --name 'Computers' --class_numbers 10 --second_category "Computers" 
```  

## Pet Supply
```shell
mkdir ~/ATG/Data/Pet/
cd ~/ATG/Data/Pet/
wget --no-check-certificate https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Pet_Supplies.json.gz
gunzip meta_Pet_Supplies.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Pet/meta_Pet_Supplies.json' --name 'Pet' --class_numbers 15  
```

```shell
mkdir ~/ATG/Data/Office/
cd ~/ATG/Data/Office/
wget --no-check-certificate https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz 
gunzip meta_Office_Products.json.gz 
cd ~/ATG/Data/
python data_processing_utils_for_books.py --data_path 'Office/meta_Office_Products.json' --name 'Office' --class_numbers 10  --second_category "0ffice & School Supplies"
```