#Import
from os import device_encoding
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
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

#load data
train_dataset = datasets.MNIST(root='dataset/', train= True, transform=transforms.ToTensor(), download=True )
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train= False, transform=transforms.ToTensor(), download=True )
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=True) 

#Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr = learning_rate)

#Train network

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to corret shape
        data = data.reshape(data.shape[0],-1)

        #forward 
        scores = model(data)
        loss = criterion(scores, targets)

        #backward 
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()


#Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking accuracy on your training data')
    else: 
        print("checking accurancy on test data")
    num_correct  = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            # 64 * 10 
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}' )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
