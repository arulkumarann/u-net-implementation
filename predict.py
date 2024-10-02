import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from model import UNet
import os

def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    processed_image = transform(image).unsqueeze(0)  # Add batch dimension

    return processed_image, original_size


def load_model(model_path):
    model = UNet()  
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])  
    model.eval()  
    return model


def make_prediction(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        return output


def save_prediction(input_image_path, model_path, output_image_path):
    input_image, original_size = preprocess_image(input_image_path)
    model = load_model(model_path)
    prediction = make_prediction(model, input_image)
    
    prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()
    
    prediction_for_binary = prediction.squeeze()
    
    threshold = 0.55 
    binary_prediction = (prediction_for_binary > threshold)
    
    binary_image = Image.fromarray(binary_prediction[0].astype(np.uint8) * 255)
    binary_image = binary_image.resize(original_size, Image.NEAREST)  # Use nearest neighbor for segmentation


    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    binary_image.save(output_image_path)

    print(f"Prediction saved at: {output_image_path}")

input_image_path = r"/teamspace/studios/this_studio/u_net_implementation/data/test/test/0a2637c772c5_03.jpg"
model_path = r"/teamspace/studios/this_studio/u_net_implementation/checkpoints/4.pth"
output_image_path = r"/teamspace/studios/this_studio/u_net_implementation/predictions/output.png"

save_prediction(input_image_path, model_path, output_image_path)
