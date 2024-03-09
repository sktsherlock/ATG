from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

url = "https://images-na.ssl-images-amazon.com/images/I/61r5kg2QfhL.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a magazine belonging to the 'Arts, Music & Photography'", "a magazine belonging to the 'Fashion & Style'"], images=image, return_tensors="pt", padding=True)

inputs1 = processor(text="a magazine belonging to the:", images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
outputs1 = model(**inputs1)

logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print('probs')
