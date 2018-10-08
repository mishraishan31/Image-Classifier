import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import iutils

ap = argparse.ArgumentParser(description='Train.py')


ap.add_argument('data_dir', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=4096)



pa = ap.parse_args()
root = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
device = pa.gpu
epochs = pa.epochs

def main():
    
    trainloader, v_loader, testloader = iutils.load_data(root)
    model, optimizer, criterion = iutils.network_construct(structure,dropout,hidden_layer1,lr,device)
    iutils.do_deep_learning(model, optimizer, criterion, epochs, 40, trainloader, device)
    iutils.save_checkpoint(model,path,structure,hidden_layer1,dropout,lr)
    print("Done Training!")


if __name__== "__main__":
    main()

