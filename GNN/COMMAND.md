# Movies
## Transductive
### Acc
#### TextFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/Transductive/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.0005 --metric=accuracy --n-epochs=1000 --n-hidden=128 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/Transductive/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.0005 --metric=accuracy --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
### F1
#### TextFeature
```python 
python GNN/Library/GCN.py --average=macro --exp_path=Exp/Transductive/ --dropout=0.2 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.0005 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py --average=macro --exp_path=Exp/Transductive/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.001 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```

## 10-Shot
### Acc
#### TextFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/10shot/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.001 --metric=accuracy --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/10shot/ --dropout=0.2 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.0005 --metric=accuracy --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
### F1
#### TextFeature
```python 
python GNN/Library/GCN.py --average=macro --exp_path=Exp/10shot/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.0005 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py --average=macro --exp_path=Exp/10shot/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.0005 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```

## 50-Shot
### Acc
#### TextFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/50shot/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=accuracy --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/50shot/ --dropout=0.2 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=accuracy --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
### F1
#### TextFeature
```python 
python GNN/Library/GCN.py --exp_path=Exp/50shot/ --average=macro --dropout=0.2 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py --exp_path=Exp/50shot/ --average=macro --dropout=0.2 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=f1 --n-epochs=1000 --n-hidden=128 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```


## Inductive
### Acc
#### TextFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/Inductive/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=accuracy --n-epochs=1000 --n-hidden=128 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py  --exp_path=Exp/Inductive/ --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=accuracy --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
### F1
#### TextFeature
```python 
python GNN/Library/GCN.py --exp_path=Exp/Inductive/ --average=macro --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/TextFeature/Movies_Llama_2_7b_hf_256_mean.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.001 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=2 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```
#### VisualFeature
```python 
python GNN/Library/GCN.py --exp_path=Exp/Inductive/ --average=macro --dropout=0.5 --early_stop_patience=100 --feature=Data/Movies/ImageFeature/Movies_openai_clip-vit-large-patch14.npy --graph_path=Data/Movies/MoviesGraph.pt  --label-smoothing=0.1 --lr=0.005 --metric=f1 --n-epochs=1000 --n-hidden=256 --n-layers=3 --n-runs=10 --selfloop=[True] --train_ratio=0.6 --undirected=[True] --val_ratio=0.2 --warmup_epochs=50
```