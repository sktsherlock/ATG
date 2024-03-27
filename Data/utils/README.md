## Amazon Fashion 
```shell
mkdir ~/ATG/Data/Fashion/
cd ~/ATG/Data/Fashion/
wget https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_AMAZON_FASHION.json.gz
gunzip meta_AMAZON_FASHION.json.gz
cd ~/ATG/Data/
python data_processing_utils.py --data_path 'Fashion/meta_AMAZON_FASHION.json' --name 'Fashion' --class_numbers 15 
``` 