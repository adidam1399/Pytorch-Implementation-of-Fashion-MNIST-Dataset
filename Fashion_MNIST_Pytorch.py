
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

# Getting the train and test data

def download_data():
    training_set=torchvision.datasets.FashionMNIST(root="./data",train=True,download=True,
                                          transform=transforms.ToTensor())
    testing_set=torchvision.datasets.FashionMNIST(root="./data",train=False,download=True,
                                          transform=transforms.ToTensor())
    return training_set, testing_set

training_set, testing_set=download_data()

# Creating the train and test loader

def data_loader(training_set, testing_set):

    # Shuffling the train data
    train_loader=torch.utils.data.DataLoader(training_set,batch_size=100,shuffle=True
                                         ,num_workers=2,pin_memory=True,
                                         drop_last=True)
    test_loader=torch.utils.data.DataLoader(testing_set,batch_size=100,shuffle=False)

    return train_loader, test_loader

train_loader, test_loader=data_loader(training_set,testing_set)

# Part 1- Model 1 for Fashion MNIST
num_of_pixels=28*28
dropout=nn.Dropout(0.2)
# Defining a neural network architecture

model=torch.nn.Sequential(
    nn.Linear(num_of_pixels,128),
    nn.ReLU(),
    nn.Linear(128,10),
    nn.Softmax(dim=1)
)

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

num_of_epochs=40
count=0
for epoch in range(num_of_epochs):
  correct=0
  for images, labels in train_loader:
    count+=1
    input_image=images.view(-1,num_of_pixels)
    outputs=model(input_image)
    loss=loss_function(outputs,labels)
    # Back prop
    optimizer.zero_grad()
    loss.backward()
    # Update weights (optimize)
    optimizer.step()
    # Evaluating the performance
    predictions=torch.max(outputs,1)[1]
    correct+=(predictions==labels).sum().numpy()

  print("Epoch is: {0}, Loss is {1} and Accuracy is: {2}".format(epoch+1,loss.data,100*correct/len(train_loader.dataset)))

print("Finished Training")


for name, param in model.named_parameters():
  print(name)

# Producing histogram of weights for the input layer for network 1

w_input = []
w_hidden = []
with torch.no_grad():
  for name, param in model.named_parameters():
    # Weights for input layer
    if name=="0.weight":
      w_input.append(param.view(-1).detach().numpy())
    # Weights for hidden layer
    if name=="2.weight":
      w_hidden.append(param.view(-1).detach().numpy())

#Weights of Input Layer
plt.hist(w_input)
plt.title('Histogram of Weights for the Input Layer of network 1')
plt.ylabel('frequency')
plt.xlabel('weight value')
plt.show()


# Producing histogram of weights for the hidden layer for network 1


plt.hist(w_hidden)
plt.title('Histogram of Weights for the hidden Layer of network 1')
plt.ylabel('frequency')
plt.xlabel('weight value')
plt.show()

# Part 2-Second model for Fashion MNIST

class Neural_Net_2(nn.Module):
  def __init__(self):
    super(Neural_Net_2,self).__init__()
    self.hidden=nn.Linear(num_of_pixels,48)
    self.output=nn.Linear(48,10)

  def forward(self,x):
    x=F.relu(self.hidden(x))
    x=dropout(x)
    x=F.softmax(self.output(x))
    return x

model_2=Neural_Net_2()


loss_function=nn.CrossEntropyLoss()
optimizer_2=torch.optim.Adam(model_2.parameters(),lr=0.001,weight_decay=0.0001)

num_of_epochs=40
count=0
for epoch in range(num_of_epochs):
  correct=0
  for images, labels in train_loader:
    count+=1
    input_image=images.view(-1,num_of_pixels)
    outputs=model_2(input_image)
    loss=loss_function(outputs,labels)
    # Back prop
    optimizer_2.zero_grad()
    loss.backward()
    # Update weights (optimize)
    optimizer_2.step()
    # Evaluating the performance
    predictions=torch.max(outputs,1)[1]
    correct+=(predictions==labels).sum().numpy()

  print("Epoch is: {0}, Loss is {1} and Accuracy is: {2}".format(epoch+1,loss.data,100*correct/len(train_loader.dataset)))

print("Finished Training")


for name, param in model_2.named_parameters():
  print(name)


# Producing histogram of weights for the input layer for network 2

w_input = []
w_hidden = []
for name, param in model_2.named_parameters():
  # Weights for input layer
  if name=="hidden.weight":
    w_input.append(param.view(-1).detach().numpy())
  # Weights for hidden layer
  if name=="output.weight":
    w_hidden.append(param.view(-1).detach().numpy())

#Weights of Input Layer
plt.hist(w_input)
plt.title('Histogram of Weights for the Input Layer of network 2')
plt.ylabel('frequency')
plt.xlabel('weight value')
plt.show()

# Producing histogram of weights for the hidden layer for network 2

plt.hist(w_hidden)
plt.title('Histogram of Weights for the hidden Layer of network 2')
plt.ylabel('frequency')
plt.xlabel('weight value')
plt.show()

# The differences between these histograms of two networks is that, for the second network, the magnitude of weights are reduced and they are more evenly spread due to regularization and dropout


