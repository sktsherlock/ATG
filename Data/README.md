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
# wget --no-check-certificate https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Electronics.jsonl.gz   Amazon 2023
gunzip meta_Electronics.json.gz   # gunzip meta_Electronics.jsonl.gz 
cd ~/ATG/Data/
python data_processing_utils_for_books.py --data_path 'Electronics/meta_Electronics.json'  --name 'Computers' --class_numbers 10 --second_category "Computers" 
# Photo
python data_processing_utils_for_books.py --data_path 'Electronics/meta_Electronics.json'  --name 'Photo' --class_numbers 12 --second_category "Photo" 
from huggingface_hub import upload_file 
upload_file(path_or_fileobj="PhotoImages.tar.gz", path_in_repo="Photo/PhotoImages.tar.gz", repo_id="Sherirto/MAG") 
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

## Office 
```shell
mkdir ~/ATG/Data/Office/
cd ~/ATG/Data/Office/
wget --no-check-certificate https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Office_Products.json.gz 
gunzip meta_Office_Products.json.gz 
cd ~/ATG/Data/
python data_processing_utils_for_books.py --data_path 'Office/meta_Office_Products.json' --name 'Office' --class_numbers 10  --second_category "Office & School Supplies"
```

# 文本图
## OGB-Arxiv
```shell
cd ~
mkdir ~/OGB
cd ~/ATG/Data/
python data_processing_for_ogb.py --data_root '/home/aiscuser/OGB/'

python TextStatisitic.py --csv_file '/home/aiscuser/ATG/Data/ogb/arxiv.csv' --column_name 'text'
```

# Download from the CS-TAG
## Books-Children and Books-Histrory
```shell
mkdir -p ~/ATG/Data/Books/Children/
cd ~/ATG/Data/Books/Children/
gdown --id 1H_7Pmfg-8o3sLNflzWOHsgL5WG01i7B6 -O ChildrenGraph.pt
gdown --id 1mERB7AF31EGHbyfvQpKholgk1cTSCKBj -O Children.csv
# Children.csv  ChildrenGraph.pt 

mkdir -p ~/ATG/Data/Books/History/
cd ~/ATG/Data/Books/History/
gdown --id 14qGkKaRAEER-huyPEJOPl9NuKtpYIInF -O HistoryGraph.pt
gdown --id 1gpBLHC6dbcpy9Ug_cvaEEzEegnRJ9dsQ -O History.csv
# History.csv  HistoryGraph.pt 
```

# 统计数据信息
```shell
# Arxiv
bash Statisitic.sh /dataintent/local/user/v-haoyan1/Data/OGB/Arxiv/OGBN_ARXIV.csv TA 512
# Movies
bash Statisitic.sh /dataintent/local/user/v-haoyan1/Data/Movies/Movies.csv text 512
# Children
bash Statisitic.sh /dataintent/local/user/v-haoyan1/Data/Books/Children/Children.csv text 512
# History
bash Statisitic.sh /dataintent/local/user/v-haoyan1/Data/Books/History/History.csv text 512
```
