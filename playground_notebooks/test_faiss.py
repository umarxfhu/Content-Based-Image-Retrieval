import os
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms, models
from sklearn.neighbors import NearestNeighbors

with open(
    "/Users/umar/UWaterloo/Musashi Winter Co-op/resources-lobe/lobe_features.pickle",
    "rb",
) as f:
    features = pickle.load(f)
with open(
    "/Users/umar/UWaterloo/Musashi Winter Co-op/resources-lobe/lobe_img_paths.pickle",
    "rb",
) as f:
    img_paths = pickle.load(f)

img_paths = [os.path.join("../src/", path) for path in img_paths]
neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean").fit(
    features
)

# Model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Deivce:", DEVICE)
model = models.resnet101(pretrained=True, progress=True)
model.to(DEVICE)

# Helper fxn
def pooling_output(x):
    global model
    for layer_name, layer in model._modules.items():
        x = layer(x)
        if layer_name == "avgpool":
            break
    return x


# Image Transform (preprocessing)
transform = transforms.Compose(
    [
        transforms.Resize(size=[224, 224], interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

my_index = 2630
query_image = img_paths[my_index]
# query_image = test_img_paths[1][0]
# print(query_image)
# print(classname(query_image))
PIL_img = Image.open(query_image)
PIL_img = PIL_img.convert("RGB")
