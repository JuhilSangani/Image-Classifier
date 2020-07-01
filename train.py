#Train your program
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

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def main():
    
    parser = argparse.ArgumentParser(description='Let\'s train Dataset')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--hidden_units', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--gpu', type=str)
    
    args, _ = parser.parse_known_args()
    data_dir = args.data_dir
    test_loaders=False
    
    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir

    arch = 'vgg19'
    if args.arch:
        arch = args.arch

    learning_rate = 0.01
    if args.learning_rate:
        learning_rate = args.learning_rate

    hidden_units = 200
    if args.hidden_units:
        hidden_units = args.hidden_units

    epochs = 5
    if args.epochs:
        epochs = args.epochs

    gpu = True
    if args.gpu:
        if torch.cuda.is_available():
            gpu = True
        else:
            print("No GPU Available!!")

    
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'training' : transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),transforms.RandomRotation(30),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                             [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]),

        'testing' : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    }
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)
    }

    class_to_idx = image_datasets['training'].class_to_idx

    dltrain = dataloaders['training']
    dltest = dataloaders['testing']
    dlvalidate = dataloaders['validation']

    if test_loaders:
        images, labels = next(iter(dltrain))
        imshow(images[2])
        plt.show()

        images, labels = next(iter(dltest))
        imshow(images[2])
        plt.show()

        images, labels = next(iter(dlvalidate))
        imshow(images[2])
        plt.show()
    
    printupto = 40
    saveupto = 50
    
#     model = models.vgg19(pretrained=True)
    model = eval("models.{}(pretrained=True)".format(arch))
        
#     classifier = nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear(25088, 4096)),
#                           ('relu', nn.ReLU()),
#                           ('fc2', nn.Linear(4096, 102)),
#                           ('output', nn.LogSoftmax(dim=1))
#                           ]))

    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, hidden_units)),
                                            ('relu', nn.ReLU()), 
                                            ('fc2', nn.Linear(hidden_units, 102)),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = classifier    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.train()
    if gpu:
        model.cuda()
    else:
        model.cpu()
        
    epochs = epochs
    step = 0

    for e in range(epochs):
        
        running_loss = 0
        accuracy_train = 0
        
        for images, labels in iter(dltrain):
            step += 1
            inputs, labels = Variable(images), Variable(labels)
            optimizer.zero_grad()
            
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ps_train = torch.exp(output).data
            equality_train = (labels.data == ps_train.max(1)[1])
            accuracy_train += equality_train.type_as(torch.FloatTensor()).mean()

            if step % printupto == 0:
                model.eval()
                accuracy = 0
                valid_loss = 0
                
                for images, labels in dlvalidate:
                    with torch.no_grad():
                        inputs = Variable(images)
                        labels = Variable(labels)

                        if gpu:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output = model.forward(inputs)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output).data
                        equality = (labels.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                print("epoch: {}/{}.. ".format(e+1, epochs), 
                      "Training Loss: {:.3f}.. ".format(running_loss / printupto),
                      "Validation Loss: {:.3f}..".format(valid_loss / len(dlvalidate)),
                      "Validation Accuracy: {:.3f}".format(accuracy / len(dlvalidate)))
                running_loss = 0
                model.train()
                
            if step % saveupto == 0:
                
                state = {'state_dict': model.classifier.state_dict(),
                         'optimizer' : optimizer.state_dict(),
                         'class_to_idx':class_to_idx}
                path = save_dir + 'checkpoint.pth.tar'
                torch.save(state, path)
                
    print(model)


# def main():

#     parser = argparse.ArgumentParser(description='Let\'s train Dataset')
#     parser.add_argument('data_dir', type=str)
#     parser.add_argument('--save_dir', type=str)
#     parser.add_argument('--arch', type=str)
#     parser.add_argument('--learning_rate', type=float)
#     parser.add_argument('--hidden_units', type=int)
#     parser.add_argument('--epochs', type=int)
#     parser.add_argument('--gpu', type=str)
    
#     args, _ = parser.parse_known_args()
#     data_dir = args.data_dir
#     test_loaders=False
    
#     save_dir = './'
#     if args.save_dir:
#         save_dir = args.save_dir

#     arch = 'densenet121'
#     if args.arch:
#         arch = args.arch

#     learning_rate = 0.1
#     if args.learning_rate:
#         learning_rate = args.learning_rate

#     hidden_units = 100
#     if args.hidden_units:
#         hidden_units = args.hidden_units

#     epochs = 2
#     if args.epochs:
#         epochs = args.epochs

#     gpu = True
#     if args.gpu:
#         if torch.cuda.is_available():
#             gpu = True
#         else:
#             print("No GPU Available!!")

#     dltrain, dltest, dlvalidate, class_to_idx = getdataloaders(data_dir)

#     if test_loaders:
#         images, labels = next(iter(dltrain))
#         imshow(images[2])
#         plt.show()

#         images, labels = next(iter(dltest))
#         imshow(images[2])
#         plt.show()

#         images, labels = next(iter(dlvalidate))
#         imshow(images[2])
#         plt.show()

#     train(dltrain, dlvalidate, class_to_idx, save_dir=save_dir, arch=arch, learning_rate=learning_rate, hidden_units=hidden_units, \
#             epochs=epochs, gpu=gpu)

if __name__ == '__main__':
    main()