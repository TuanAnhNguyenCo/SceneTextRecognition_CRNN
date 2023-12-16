import os
import shutil
import xml.etree.ElementTree as ET 
import numpy as np
import pandas as pd
import random
import cv2
import timm
import matplotlib.pyplot as plt 
from PIL import Image

from opts import parse_opts_offline
from prepare_data_yolo_v8 import extract_data_from_xml


def split_bounding_boxes(img_paths , img_labels , bboxes,opts):
    os.makedirs(opts.crnn_data_dir , exist_ok=True)
    
    count = 0
    labels = [] 
    for img_path , img_labels , bbs in zip(img_paths , img_labels , bboxes): 
        img = Image.open(os.path.join(opts.root_path,img_path))
        for label,bb in zip(img_labels,bbs):
            cropped_image = img.crop((bb[0],bb[1],bb[0]+bb[2],bb[1]+bb[3]))
            # if 90% of the image is black or white, it will be ignored
            if np.mean(cropped_image) < 35 and np.mean(cropped_image) > 220:
                continue
            # if the size of the image is too small (maybe in the corner of the image and it's shortage) , it will be ignored
            if cropped_image.size[0] < 10 or cropped_image.size[1] < 10: 
                continue
            # Save image
            filename = f"{count:06d}.jpg" 
            cropped_image.save(os.path.join(opts.crnn_data_dir , filename))
            new_img_path = os.path.join(opts.crnn_data_dir , filename)
            label = new_img_path + '\t' + label
            labels.append(label)
            count += 1
        
        
    print(f"Created {count} images")
    # Write labels to a text file
    with open(os.path.join(opts.crnn_data_dir , 'labels.txt'),'w') as f:
        for label in labels:
            f.write(f"{label}\n")
            
if __name__ == '__main__': 
    opts = parse_opts_offline()
    img_paths , img_sizes , img_labels , bboxes = extract_data_from_xml(opts)
    split_bounding_boxes(img_paths,img_labels,bboxes,opts)
    
            





