# Movies 
## GCN 
### Transductive
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric accuracy --text_logits Exp/Transductive/Movies/GCN/TextFeature/  --visual_logits Exp/Transductive/Movies/GCN/ImageFeature/
```
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric f1 --average macro --text_logits Exp/Transductive/Movies/GCN/TextFeature/   --visual_logits Exp/Transductive/Movies/GCN/ImageFeature/
```
### Inductive  
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric accuracy --text_logits Exp/Inductive/Movies/GCN/TextFeature/  --visual_logits Exp/Inductive/Movies/GCN/ImageFeature/
```
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric f1 --average macro --text_logits Exp/Inductive/Movies/GCN/TextFeature/   --visual_logits Exp/Inductive/Movies/GCN/ImageFeature/
```

### 10-Shot
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric accuracy --text_logits Exp/10shot/Movies/GCN/TextFeature/  --visual_logits Exp/10shot/Movies/GCN/ImageFeature/
```
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric f1 --average macro --text_logits Exp/10shot/Movies/GCN/TextFeature/   --visual_logits Exp/10shot/Movies/GCN/ImageFeature/
```
### 50-Shot  
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric accuracy --text_logits Exp/50shot/Movies/GCN/TextFeature/  --visual_logits Exp/50shot/Movies/GCN/ImageFeature/
```
```python
python Ensemble.py --graph_path Data/Movies/MoviesGraph.pt --metric f1 --average macro --text_logits Exp/50shot/Movies/GCN/TextFeature/   --visual_logits Exp/50shot/Movies/GCN/ImageFeature/
```


