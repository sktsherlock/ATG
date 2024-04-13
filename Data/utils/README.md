## Amazon Fashion 
```shell
mkdir ~/ATG/Data/Fashion/
cd ~/ATG/Data/Fashion/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_AMAZON_FASHION.json.gz
gunzip meta_AMAZON_FASHION.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Fashion/meta_AMAZON_FASHION.json' --name 'Fashion' --class_numbers 15 
``` 

## Arts 可能可以 试试 
## 看看二级 Sewing
python data_processing_utils_for_books.py --data_path 'Arts/meta_Arts_Crafts_and_Sewing.json'  --name 'Sewing' --class_numbers 15 --second_category "Sewing" 
```shell 
mkdir ~/ATG/Data/Arts/
cd ~/ATG/Data/Arts/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Arts_Crafts_and_Sewing.json.gz
gunzip meta_Arts_Crafts_and_Sewing.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Arts/meta_Arts_Crafts_and_Sewing.json' --name 'Arts' --class_numbers 11 --save --download_image
``` 

```shell 
mkdir ~/ATG/Data/AutoMotive/
cd ~/ATG/Data/AutoMotive/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Automotive.json.gz
gunzip meta_Automotive.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'AutoMotive/meta_Automotive.json' --name 'AutoMotive' --class_numbers 11 --save --download_image
``` 

```shell 
mkdir ~/ATG/Data/Grocery/
cd ~/ATG/Data/Grocery/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Grocery_and_Gourmet_Food.json.gz
gunzip meta_Grocery_and_Gourmet_Food.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Grocery/meta_Grocery_and_Gourmet_Food.json' --name 'Grocery' --class_numbers 20 --save --download_image
python data_processing_utils.py --data_path 'Grocery/meta_Grocery_and_Gourmet_Food.json' --name 'Grocery_5000' --class_numbers 20 --save --download_image --sampling 5000
``` 



```shell 
mkdir ~/ATG/Data/Toys/
cd ~/ATG/Data/Toys/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Toys_and_Games.json.gz
gunzip meta_Toys_and_Games.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Toys/meta_Toys_and_Games.json' --name 'Toys' --class_numbers 20  --save --download_image
``` 