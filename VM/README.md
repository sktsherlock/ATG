# Ready for the folder type datasets
```bash
cd ~/ATG/Data/
```
```python
python utils/Create_split.py --csv_path RedditS/RedditS.csv   --graph_path RedditS/RedditSGraph.pt  --photos_path RedditS/RedditSImages/ --save_path RedditS/ImageTask/  
```
```bash
cd ~/ATG/
pip install torchmetrics
pip install pytorch_accelerated
```
```python
CUDA_VISIBLE_DEVICES=2 python VM/ImageClassification.py --data_path Data/RedditS/ImageTask/  --model_name timm/vit_base_patch16_clip_224.openai --image_size 224
```