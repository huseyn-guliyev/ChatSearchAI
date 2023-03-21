import numpy as np
import clip
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import os

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
def get_model():
  model, _ = clip.load("ViT-B/32", device = device)

  return model

def get_preprocess():
  _, preprocess = clip.load("ViT-B/32", device= device)

  return preprocess
model, preprocess = get_model(), get_preprocess()
model.eval()

# Load and preprocess the images
image_paths = list(map(lambda x: './images/' + x, os.listdir('./images')))
images = [Image.open(path).convert("RGB") for path in image_paths]
images = [preprocess(image).unsqueeze(0).to(device) for image in images]

# Extract image features and concatenate them into a single array
image_features = []
for image in tqdm(images, total=len(images)):
    with torch.no_grad():
        features = model.encode_image(image).cpu().numpy()
    image_features.append(features)
image_features = np.concatenate(image_features, axis=0)

# Save the image features as a npy file
np.save("image_features.npy", image_features)
