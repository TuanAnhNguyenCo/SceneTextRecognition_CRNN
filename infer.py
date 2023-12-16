import os
import numpy as np
import timm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models 
from torchvision import transforms 
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from utils import decode,build_vocab,read_data
from opts import parse_opts_offline
from crnn import CRNN
import matplotlib.patches as patches
import torch
from torch import nn
import random

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((100, 420)), 
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))])
    
}
def text_detection(img_path , text_det_model):
    text_det_results = text_det_model(img_path , verbose=False)[0]
   
    
    bboxes = text_det_results.boxes.xyxy.tolist() 
    classes = text_det_results.boxes.cls.tolist() 
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()
    return bboxes , classes , names , confs
def text_recognition(img, data_transforms, text_reg_model, idx_to_char, device):
    transformed_image = data_transforms(img)
    transformed_image = transformed_image.unsqueeze(0).to(device) 
    text_reg_model.eval()
    with torch.no_grad():
        logits = text_reg_model(transformed_image).detach().cpu() 
        text = decode(logits.permute(1, 0, 2).argmax(2), idx_to_char)
    return text

def visualize_detections(img, detections,thread_hold):
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for bbox, detected_class, confidence, transcribed_text in detections:
        if confidence < thread_hold:
            continue
        x1, y1, x2, y2 = bbox

        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
        
        # Add the rectangle to the axis
        ax.add_patch(rect)
        
        # Add text to the image
        plt.text(x1, y1 - 10, f"({transcribed_text}) ", fontsize=9, bbox=dict(facecolor='red', alpha=0.5))

    # Save the figure with the detections
    plt.savefig('result.jpeg')
    
def predict(img_path , data_transforms , text_det_model , text_reg_model , idx_to_char , device,thread_hold):
    # Detection
    bboxes, classes, names, confs = text_detection(img_path, text_det_model) # Load the image
    img = Image.open(img_path) 
    predictions = []
    # Iterate through the results
    for bbox, cls, conf in zip(bboxes, classes, confs): 
        x1, y1, x2, y2 = bbox
        confidence = conf
        detected_class = cls
        name = names[int(cls)]
        # Extract the detected object and crop it
        cropped_image = img.crop((x1, y1, x2, y2))
        transcribed_text = text_recognition( cropped_image ,data_transforms , text_reg_model , idx_to_char , device)
        predictions.append((bbox, name, confidence, transcribed_text)) 
    visualize_detections(img, predictions,thread_hold)
    return predictions
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
def infer(img_path):
    seed_everything(20)
    opts = parse_opts_offline()
    chars = '0123456789abcdefghijklmnopqrstuvwxyz-'
    vocab_size = len(chars)
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))} 
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
    text_det_model_path = 'models/yolov8/detect/train/weights/best.pt'
    yolo = YOLO(text_det_model_path).to(opts.device)
    crnn_model = CRNN(vocab_size=vocab_size , hidden_size=opts.hidden_size , n_layers=opts.n_layers , 
                 dropout=opts.dropout_prob , unfreeze_layers=opts.unfreeze_layers).to(opts.device)
    crnn_model.load_state_dict(torch.load("models/model/crnn.pt"))
    predict(img_path,data_transforms['val'],yolo,crnn_model,idx_to_char,opts.device,opts.threadhold)
    

if __name__ == "__main__":
    infer('demo.jpeg')
