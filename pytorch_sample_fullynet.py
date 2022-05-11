#Import
import torch
import torch.nn as nn  #all linear nural network modules , convulution network is also in nn
import torch.optim as optim # all the optimism algorithm eg.Adam
import torch.nn.functional as F # all funtions that don't have parameters eg.relu,tann
from torch.utils.data import DataLoader # give you easiler dataset management. eg mini batch 
import torchvision.datasets as datasets  # get dataset from 
import torchvision.transforms as transforms  # tranformation that we can perform the data set 


#Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # (28X28) = 784
        super(NN,self).__init__()  #run initialization of the parents module. in this case, nn.module
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self ,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

model = NN(784,10)
x = torch.randn(64,784) #initialize x <- number of images (mini batch size)
print(model(x).shape)

#Set device 
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size = 874
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root='dataset/', train= True, transform=transforms.ToTensor(), download=True )
trainer_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train= True, transform=transforms.ToTensor(), download=True )
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True) 
##10분 