from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
print(model)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
print(outputs)

last_hidden_state = outputs.last_hidden_state
print(last_hidden_state.shape)