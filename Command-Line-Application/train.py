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
import argparse
import helper

parser = argparse.ArgumentParser(description='Image Classifier Trainer')

parser.add_argument('data_dir', action="store", default="flowers")
parser.add_argument('--save_dir', action="store", default="")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float, default="0.01")
parser.add_argument('--hidden_unit', action="store", type=int, default=4096)
parser.add_argument('--epochs', action="store", type=int, default="7")
parser.add_argument('--gpu', action="store_true", default=True)

args = parser.parse_args()

arch = args.arch.lower()

# Add as many archs as needed

if arch!="vgg16" and arch!="alexnet":
    print("Sorry, Architecture Not Supported")
    exit()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

############################
# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

data_transforms = {'train' : train_transforms, 'valid' : valid_transforms, 'test' : test_transforms}

# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

image_datasets = {'train' : train_data, 'valid' : valid_data, 'test' : test_data}

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32)
testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)

dataloaders = {'train' : trainloader, 'valid' : validloader, 'test' : testloader}
############################
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
############################
model = models.__dict__[arch](pretrained=True)

############################
for param in model.parameters():
    param.requires_grad = False
############################
# HyperParameter
if arch=="vgg16":
    in_features = 25088
elif arch=="alexnet":
    in_features = 9216

hidden_layer = args.hidden_unit
out_features = 102
dropout = 0.5
epochs = args.epochs
learning_rate = args.learning_rate
print_every = 40
steps = 0

# Network
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_layer)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_layer, 1000)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(dropout)),
                          ('fc3', nn.Linear(1000, out_features)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier
device = torch.device("cpu")
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
model.to(device)
############################
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
# Training
for e in range(epochs):
    running_loss = 0
    running_accuracy = 0
    for ii, (inputs, labels) in enumerate(dataloaders['train']):

        steps += 1
        
        # Move input and label tensors to the GPU
        
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        #print(model.classifier.parameters)
        
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        running_accuracy += equality.type(torch.FloatTensor).mean()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, valid_accuracy = helper.validation(model, dataloaders['valid'], criterion)

            print("Epoch: {}/{}... ".format(e+1, epochs),
                "Train Loss: {:.4f}".format(running_loss/print_every),
                "Train Accuracy: {:.4f}".format(running_accuracy/print_every),
                "Validation Loss: {:.4f}".format(valid_loss/len(dataloaders['valid'])),
                "Validation Accuracy: {:.4f}".format(valid_accuracy/len(dataloaders['valid'])))
            
            running_loss = 0
############################
# TODO: Do validation on the test set
with torch.no_grad():
    test_loss, test_accuracy = helper.validation(model, dataloaders['test'], criterion)

print("Test Loss: {:.4f}".format(test_loss/len(dataloaders['test'])),
      "Test Accuracy: {:.4f}".format(test_accuracy/len(dataloaders['test'])))
############################
# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {
    'class_to_idx' : model.class_to_idx,
    'classifier' : model.classifier,
    'epochs' : epochs,
    'arch' : arch,
    'learning_rate' : learning_rate
}
torch.save(checkpoint, args.save_dir + 'checkpoint.pth')
print('Model Saved Successfully')