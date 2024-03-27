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
```shell 
mkdir ~/ATG/Data/Arts/
cd ~/ATG/Data/Arts/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Arts_Crafts_and_Sewing.json.gz
gunzip meta_Arts_Crafts_and_Sewing.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Arts/meta_Arts_Crafts_and_Sewing.json' --name 'Arts' --class_numbers 15 
``` 

```shell 
mkdir ~/ATG/Data/AutoMotive/
cd ~/ATG/Data/AutoMotive/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_Automotive.json.gz
gunzip meta_Automotive.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'AutoMotive/meta_Automotive.json' --name 'AutoMotive' --class_numbers 15 
``` 