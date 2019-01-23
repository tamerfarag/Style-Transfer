import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json
from collections import OrderedDict
import time
import random, os
from PIL import Image
import numpy as np

def validation(model, testloader, criterion, gpu=True):
    test_loss = 0
    accuracy = 0
    device = torch.device("cpu")
    if gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    for inputs, labels in testloader:

        # Move input and label tensors to the GPU
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True) 
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']   
    arch = checkpoint['arch']   
    learning_rate = checkpoint['learning_rate']    
       
    return model, arch, learning_rate

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
  
    minwidth = 256
    minheight = 256
        
    wpercent = (minwidth/float(image.size[0]))
    hpercent = (minheight/float(image.size[1]))    
    
    wsize = int((float(image.size[0])*float(hpercent)))
    hsize = int((float(image.size[1])*float(wpercent)))
    
    if wsize<minwidth:
        image = image.resize((minwidth, hsize), Image.ANTIALIAS)
    if hsize<minheight:
        image = image.resize((wsize, minheight), Image.ANTIALIAS)
        
    width, height = image.size
    
    left = int((width - 224)/2)
    if left<0:
        left=0

    top = int((height - 224)/2)
    if top<0:
        top=0

    right = left + 224
    if width<224:
        right=width

    bottom = top + 224
    if height<224:
        bottom=height

    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    norm_image = (np_image/255-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    trans_image = norm_image.transpose((2,0,1))
    return trans_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5, gpu=True, cat_names='cat_to_name.json'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file

    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
    img = Image.open(image_path)
    img = process_image(img)

    device = torch.device("cpu")
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.to(device)    
    #model.cpu()
    inputs = torch.FloatTensor(img).unsqueeze_(0)
    print(inputs.shape)
    model.eval()
    with torch.no_grad():
        if gpu and torch.cuda.is_available():
            output = model.forward(inputs.float().cuda())
        else:
            output = model.forward(inputs.float())
        props, labels_ids = torch.topk(output, topk)
    probs = props.exp()
    class_to_idx = {model.class_to_idx[k]: k for k in model.class_to_idx}
    labels = []
    for x in np.array(labels_ids)[0]:
        labels.append(cat_to_name[class_to_idx[x]])
    return np.array(props)[0], labels
