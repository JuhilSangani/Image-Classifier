from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import shutil
import argparse

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
#     print(checkpoint)
    model = models.densenet121(pretrained=True)
#     model = eval("models.{}(pretrained=True)".format(arch))
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, 200)),
                                            ('relu', nn.ReLU()), 
                                            ('fc2', nn.Linear(200, 102)),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
    
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    
#     classifier = nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear(25088, 4096)),
#                           ('relu', nn.ReLU()),
#                           ('fc2', nn.Linear(4096, 102)),
#                           ('output', nn.LogSoftmax(dim=1))
#                           ]))

    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    adjust = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    img_tensor = adjust(img)
    
    return img_tensor

def predict(image_path, checkpoint, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_model(checkpoint)
#     if gpu:
#         model.cuda()
#     else:
#         model.cpu()
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk)
    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()
    top_labels = []
    
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

def main():

    parser = argparse.ArgumentParser(description='Predict name')
    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--category_names', type=str)
    parser.add_argument('--gpu', type=str)

    args, _ = parser.parse_known_args()
    image_path = args.input
    checkpoint = args.checkpoint

    top_k = 1
    if args.top_k:
        top_k = args.top_k

    category_names = None
    if args.category_names:
        category_names = args.category_names

    gpu = True
    if args.gpu:
        if torch.cuda.is_available():
            gpu = True
        else:
            print("No GPU Available!!")

    probs, classes, class_names = predict(image_path, checkpoint, topk=top_k)

    print('FLOWER PREDICTOR')
    print("Input label = {}".format(classes))
    print("Probability = {}".format(probs))
    print("Classnames= {}".format(class_names))

if __name__ == '__main__':
    main()