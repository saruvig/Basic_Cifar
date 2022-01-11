# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import numpy as np

DATA_ROOT = "./dataset"
DEVICE = torch.device("cpu")

class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  #3 input image channel, 6 output channels, 5x5 sq convolution
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) #6
        #affine operation y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_cifar_data():
    #CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    #CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    CIFAR10_MEAN = (0.485, 0.456, 0.406)
    CIFAR10_STD_DEV = (0.229,0.224, 0.225)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),])

    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform)
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=False)
    
    return trainloader, testloader


def train(net, trainloader, optimizer, epochs, DEVICE):
    net.to(DEVICE)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()

    for epch in range(epochs): 
        correctTrain, totalTrain, lossTrain = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            lossTrain += loss.item()

            loss.backward()
            optimizer.step()
            
            totalTrain += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correctTrain += (predicted==labels).sum().item()
            
        accuracyTrain = correctTrain / totalTrain
        
        print(f"Train {epch} - [{(len(trainloader))};{(len(trainloader.dataset))}]. Loss:{lossTrain:.3f},AccTrain [{correctTrain}/{totalTrain}]:{accuracyTrain:.3f}")

    return lossTrain, accuracyTrain

def test(net, dataloader, DEVICE):
    """Evaluate the network on the input dataset."""
    #define loss and metrics
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    # Evaluate the network
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
        
    return loss, accuracy


def main():
    print("Load data")
    trainloader, testloader = load_cifar_data()

    # Initialize training
    net = Net().to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epoch = 20
    
    print("Training...")
    train_loss , train_accuracy = train(net, trainloader, optimizer, epoch, device=DEVICE)
    loss, accuracy = test(net, trainloader, DEVICE=DEVICE)
    print("Final Train Accuracy: {}, Final Train Loss: {}".format(accuracy, loss))

    print("Evaluate on test set")
    loss, accuracy = test(net, testloader, DEVICE=DEVICE)
    print("Test Accuracy: {}, Test Loss: {}".format(accuracy, loss))

if __name__ == "__main__":
    main()