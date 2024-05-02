import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import json

# Load pre-trained model
model = models.resnet18(weights='IMAGENET1K_V1')
# Remove last fully connected layer
model = torch.nn.Sequential(*list(model.children())[:-1])
# Set model to evaluation mode
model.eval()

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def extract_embedding(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze().numpy()

if __name__ == "__main__":

    # Build database of embeddings
    embeddings_dict = {}
    image_folder = "data/slide_dataset_processed_images/dataset_presentation_1_processed"
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            embedding = extract_embedding(image_path)
            embeddings_dict[filename] = embedding.tolist()

    with open("embeddings_dict.json", 'w') as f:
        json.dump(embeddings_dict, f)