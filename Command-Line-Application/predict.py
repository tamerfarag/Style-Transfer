#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

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


parser = argparse.ArgumentParser(description='Image Classifier Predictor')

parser.add_argument('input', action="store", default="flowers/train/43/image_02379.jpg")
parser.add_argument('checkpoint', action="store", default="checkpoint.pth")
parser.add_argument('--top_k', action="store", default="3", type=int)
parser.add_argument('--category_names', action="store", default="cat_to_name.json")
parser.add_argument('--gpu', action="store_true", default=True)

args = parser.parse_args()


############################

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

plt.rcdefaults()
img_path = args.input
model, _, _ = helper.load_checkpoint(args.checkpoint)
probs, labels = helper.predict(img_path, model, topk = args.top_k, gpu = args.gpu)
segments = args.input.split('/')
print(cat_to_name[str(segments[2])])
print (probs, labels)

############################
