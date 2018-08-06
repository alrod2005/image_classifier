import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json
from util import generate_model, build_classifier
import argparse

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = generate_model(checkpoint['arch'])
    classifier = build_classifier(checkpoint['input_size'],
                             checkpoint['hidden_units'],
                             checkpoint['output_size'],
                             checkpoint['dropout'])
    classifier.load_state_dict(checkpoint['state_dict'])
    model.classifier = classifier
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    
    # Resize
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int(height * float(new_width) / width)
    else:
        new_height = 256
        new_width = int(width * float(new_height) / height)
    
    image = image.resize((new_width, new_height))
    
    # Crop
    left = (new_width - 224) / 2
    right = new_width - (new_width - 224) / 2
    upper = (new_height - 224) / 2
    lower = new_height - (new_height - 224) / 2
    image = image.crop((left, upper, right, lower))
    
    # Normalize
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = np.array(image) / 255.0
    np_image = (np_image - means) /  stds
    
    # Transpose
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, device, top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)
    
    # Predict the class from an image file
    processed_image = process_image(image_path)
    image_tensor = torch.from_numpy(np.expand_dims(processed_image, axis=0)).float()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probs, labels = output.topk(top_k)
    probs = np.array(probs.exp().data)[0]
    labels = np.array(labels)[0]
        
    return probs, labels

def main(image_path, checkpoint, top_k, category_names, gpu):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_checkpoint(checkpoint)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu) else "cpu")
    
    probs, labels = predict(image_path, model, device, top_k)
    
    label_map = {v: k for k, v in model.class_to_idx.items()}
    classes = [cat_to_name[label_map[l]] for l in labels]
    
    return classes, probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict an image class using saved, trained classifier.',
    )
    
    parser.add_argument('image_path', default='flowers/train/1/image_06734.jpg')
    parser.add_argument('checkpoint', default='checkpoint.pth')
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    input_args = parser.parse_args()

    classes, probs = main(input_args.image_path, input_args.checkpoint, input_args.top_k,
                          input_args.category_names, input_args.gpu)
    
    print([x for x in zip(classes, probs)])