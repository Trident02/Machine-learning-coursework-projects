# NAME : Shiva Ganesh Reddy Lakkasani
# ROLL NUMBER : 20EE10069
# MLFA (AI42001) Assignment - 7

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Defining plotting settings
plt.rcParams['figure.figsize'] = 14, 6

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

# Initializing normalizing transform for the dataset
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std)])

# Downloading the CIFAR10 dataset into train and test sets
train_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/train", train=True,
    transform=normalize_transform,
    download=True)

test_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/test", train=False,
    transform=normalize_transform,
    download=True)

# Generating data loaders from the corresponding datasets
batch_size = 256
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)



#########
#########
#########
#########

### CNNVanilla with Residual connection removed from the CNNResnet  :  ###

# Define the Modified Residual Block without the residual connection 
class ModifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ModifiedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

# Define the CNN-Vanilla
class CNNVanilla(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CNNVanilla, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # Use the modified residual blocks which don't have skip connections
        self.res1 = ModifiedResidualBlock(32, 32)
        self.res2 = ModifiedResidualBlock(32, 32)
        self.res3 = ModifiedResidualBlock(32, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model_vanilla = CNNVanilla().to(device)

# # Printing model summary
# print(model_vanilla)
# summary(model_vanilla, (3, 32, 32))



#########
#########
#########
#########


### CNNResnet with ADAM Optimizer, Data Normalization : ###


# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

# Define the CNN-Resnet
class CNNResnet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CNNResnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 32)
        self.res3 = ResidualBlock(32, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model_resnet = CNNResnet().to(device)

# # Printing model summary
# print(model_resnet)
# summary(model_resnet, (3, 32, 32))





#########
#########
#########
#########

## EXPERIMENT - 1 : ##


## PART - A : TRAINING CNNVanilla ##


# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_vanilla.parameters(), lr=learning_rate)

# Training process begins
train_loss_list_vanilla = []
train_accuracy_list_vanilla = []
print(f"[EXPT - 1] : Training CNNVanilla \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_vanilla.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_vanilla(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_vanilla.append(train_loss / len(train_loader))

    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_vanilla.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_vanilla[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")


## Testing CNNVanilla ###
test_acc_vanilla =0
model_vanilla.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_vanilla(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_vanilla += (y_pred == y_true).sum().item()

    print(f"[EXPT - 1] : Test set accuracy with CNNVanilla = {100 * test_acc_vanilla / len(test_dataset)} %")



#########
#########

## PART - B : TRAINING CNNResnet with ADAM Optimizer, Data Normalization ##

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet.parameters(), lr=learning_rate)

# Training process begins
train_loss_list_resnet = []
train_accuracy_list_resnet = []
print(f"[EXPT - 1] : Training CNNResnet \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_resnet.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_resnet(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_resnet.append(train_loss / len(train_loader))

    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_resnet.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_resnet[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")

## Testing CNNResnet ###
test_acc_resnet =0
model_resnet.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_resnet += (y_pred == y_true).sum().item()

    print(f"[EXPT - 1] : Test set accuracy with CNNResnet = {100 * test_acc_resnet / len(test_dataset)} %")




# Plotting Training Accuracy Vs Epochs CNN Vanilla and CNN Resnet
plt.figure(figsize=(14, 6))
plt.plot(range(1, num_epochs + 1), train_accuracy_list_vanilla, label='With CNN Vanilla')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet, label='With CNN Resnet')
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")
plt.title("[EXPT - 1] : Training Accuracy Vs Epochs CNN Vanilla and CNN Resnet")
plt.legend()
plt.show()

#########
#########
#########
#########

## EXPERIMENT - 2 : CNNResnet without Data Normalization ##

# Transformation without normalization
transform_no_normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Downloading the CIFAR10 dataset without normalization
train_dataset_no_norm = torchvision.datasets.CIFAR10(
    root="./CIFAR10/train", train=True,
    transform=transform_no_normalize,
    download=True)

test_dataset_no_norm = torchvision.datasets.CIFAR10(
    root="./CIFAR10/test", train=False,
    transform=transform_no_normalize,
    download=True)

# Data loaders for the datasets without normalization
train_loader_no_norm = torch.utils.data.DataLoader(
    train_dataset_no_norm, shuffle=True, batch_size=batch_size)
test_loader_no_norm = torch.utils.data.DataLoader(
    test_dataset_no_norm, batch_size=batch_size)


# Initialize the model
model_resnet_no_norm = CNNResnet().to(device)

# Define the loss criterion and optimizer for the new model
num_epochs = 50
learning_rate = 0.001
criterion_no_norm = torch.nn.CrossEntropyLoss()
optimizer_no_norm = torch.optim.Adam(model_resnet_no_norm.parameters(), lr=learning_rate)

# Train the model without normalization
train_loss_list_resnet_no_norm = []
train_accuracy_list_resnet_no_norm = []
print(f"[EXPT - 2] : Training CNNResnet without Data Normalization \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs} (No Normalization):', end=' ')
    train_loss_no_norm = 0
    correct_no_norm = 0
    total_no_norm = 0

    # Switch to training mode
    model_resnet_no_norm.train()

    # Training loop
    for i, (images, labels) in enumerate(train_loader_no_norm):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs_no_norm = model_resnet_no_norm(images)
        loss_no_norm = criterion_no_norm(outputs_no_norm, labels)

        # Backward pass and optimize
        optimizer_no_norm.zero_grad()
        loss_no_norm.backward()
        optimizer_no_norm.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs_no_norm.data, 1)
        total_no_norm += labels.size(0)
        correct_no_norm += (predicted == labels).sum().item()

        train_loss_no_norm += loss_no_norm.item()

    # Record the training loss
    train_loss_list_resnet_no_norm.append(train_loss_no_norm / len(train_loader_no_norm))
    
    # Average accuracy for this epoch
    epoch_accuracy_no_norm = 100 * correct_no_norm / total_no_norm
    train_accuracy_list_resnet_no_norm.append(epoch_accuracy_no_norm)

    print(f"Training loss = {train_loss_list_resnet_no_norm[-1]}, Training Accuracy = {epoch_accuracy_no_norm:.2f}%")


## Testing CNNResnet without Data Normalization ###
test_acc_resnet_no_norm =0
model_resnet_no_norm.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader_no_norm):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet_no_norm(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_resnet_no_norm += (y_pred == y_true).sum().item()

    print(f"[EXPT - 2] : Test set accuracy for CNNResnet without Data Normalization = {100 * test_acc_resnet_no_norm / len(test_dataset_no_norm)} %")



# Plot the training loss for CNNResnet with and without normalization
plt.figure(figsize=(14, 6))
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet, label='CNNResnet With Normalization')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet_no_norm, label='CNNResnet Without Normalization')
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")
plt.title("[EXPT - 2] : Training Accuracy Vs Epochs for CNNResnet with and without Data Normalization")
plt.legend()
plt.show()


#########
#########
#########
#########

## EXPERIMENT - 3 : ##

## Using Different Optimizers for CNN Resnet with Data Normalization ## : 

#########
#########

## PART - A : TRAINING CNNResnet with SGD Optimizer, Data Normalization ##

model_resnet_sgd = CNNResnet().to(device)

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet_sgd.parameters(), lr=0.001)

# Training process begins
train_loss_list_resnet_sgd = []
train_accuracy_list_resnet_sgd = []
print(f"\n [EXPT - 3, PART - A ] : Training CNNResnet with SGD Optimizer, Data Normalization  \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_resnet_sgd.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_resnet_sgd(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_resnet_sgd.append(train_loss / len(train_loader))
    
    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_resnet_sgd.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_resnet_sgd[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")


## Testing CNNResnet with SGD Optimizer, Data Normalization  ###
test_acc_sgd =0
model_resnet_sgd.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet_sgd(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_sgd += (y_pred == y_true).sum().item()

    print(f"[EXPT - 3, PART - A] : Test set accuracy with CNNResnet with SGD Optimizer, Data Normalization = {100 * test_acc_sgd / len(test_dataset)} %")


#########
#########


## PART - B : TRAINING CNNResnet with Mini-Batch Gradient Descent with no momentum, Data Normalization ##

model_resnet_mini_batch_no_momentum = CNNResnet().to(device)

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet_mini_batch_no_momentum.parameters(), lr=0.001, momentum=0)

# Training process begins
train_loss_list_resnet_mini_batch_no_momentum = []
train_accuracy_list_resnet_mini_batch_no_momentum = []
print(f"\n [EXPT - 3, PART - B] : Training CNNResnet with Mini-Batch Gradient Descent with no momentum, Data Normalization  \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_resnet_mini_batch_no_momentum.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_resnet_mini_batch_no_momentum(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_resnet_mini_batch_no_momentum.append(train_loss / len(train_loader))
    
    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_resnet_mini_batch_no_momentum.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_resnet_mini_batch_no_momentum[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")


#########
#########


## Testing CNNResnet with Mini-Batch Gradient Descent with no momentum, Data Normalization ###
test_acc_mini_batch_no_momentum = 0
model_resnet_mini_batch_no_momentum.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet_mini_batch_no_momentum(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_mini_batch_no_momentum += (y_pred == y_true).sum().item()

    print(f"[EXPT - 3, PART - B ] : Test set accuracy with CNNResnet with Mini-Batch Gradient Descent with no momentum, Data Normalization = {100 * test_acc_mini_batch_no_momentum / len(test_dataset)} %")


#########
#########


## PART - C : TRAINING CNNResnet with Mini-Batch Gradient Descent with momentum 0.9, Data Normalization ##

model_resnet_mini_batch_with_momentum = CNNResnet().to(device)

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet_mini_batch_with_momentum.parameters(), lr=0.001, momentum=0.9)

# Training process begins
train_loss_list_resnet_mini_batch_with_momentum = []
train_accuracy_list_resnet_mini_batch_with_momentum = []
print(f"\n [EXPT - 3, PART - C] : Training CNNResnet with Mini-Batch Gradient Descent with momentum 0.9, Data Normalization  \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_resnet_mini_batch_with_momentum.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_resnet_mini_batch_with_momentum(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_resnet_mini_batch_with_momentum.append(train_loss / len(train_loader))
    
    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_resnet_mini_batch_with_momentum.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_resnet_mini_batch_with_momentum[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")


#########
#########


## Testing CNNResnet with Mini-Batch Gradient Descent with momentum 0.9, Data Normalization ###
test_acc_mini_batch_with_momentum =0
model_resnet_mini_batch_with_momentum.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet_mini_batch_with_momentum(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_mini_batch_with_momentum += (y_pred == y_true).sum().item()

    print(f"[EXPT - 3, PART - C ] : Test set accuracy with CNNResnet with Mini-Batch Gradient Descent with momentum 0.9, Data Normalization = {100 * test_acc_mini_batch_with_momentum / len(test_dataset)} %")



#########
#########


## PART - D : TRAINING CNNResnet with ADAM Optimizer ##

## ALREADY DONE IN EXPERIMENT - 1 ##

#########
#########

## PLOTS ##

plt.figure(figsize=(14, 6))
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet, label='With CNN Resnet with ADAM Optimizer')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet_sgd, label='With CNN Resnet with SGD Optimizer')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet_mini_batch_no_momentum, label='With CNN Resnet with Mini-Batch Gradient Descent with no momentum')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet_mini_batch_with_momentum, label='With CNN Resnet with Mini-Batch Gradient Descent with momentum 0.9')
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")
plt.title("[EXPT - 3] : Training Accuracy Vs Epochs for CNN Resnet with Different Optimizer")
plt.legend()
plt.show()



#########
#########
#########
#########


## EXPERIMENT - 4 : ##

# PART - A #

# Four level Resnet block with two fully-connected layers
class CNNResnet_DeepConv(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CNNResnet_DeepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 32)
        self.res3 = ResidualBlock(32, 32)
        self.res4 = ResidualBlock(32, 32)  # additional ResNet block
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 512)  # adjusted to match the correct flattened size
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)  # Additional ResNet block
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model_resnet_deep_conv = CNNResnet_DeepConv().to(device)


# # Printing model summary
# print(model_resnet_deep_conv)
# summary(model_resnet_deep_conv, (3, 32, 32))

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet_deep_conv.parameters(), lr=learning_rate)

# Training process begins
train_loss_list_resnet_deep_conv = []
train_accuracy_list_resnet_deep_conv = []
print(f"\n [EXPT - 4] : Training CNNResnet  - Four level Resnet block with two fully-connected layer\n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_resnet_deep_conv.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_resnet_deep_conv(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_resnet_deep_conv.append(train_loss / len(train_loader))
    
    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_resnet_deep_conv.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_resnet_deep_conv[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")


## Testing CNNResnet  - Four level Resnet block with two fully-connected layer ###
test_acc_resnet_deep_conv =0
model_resnet_deep_conv.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet_deep_conv(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_resnet_deep_conv += (y_pred == y_true).sum().item()

    print(f"\n [EXPT - 4] : Test set accuracy with CNNResnet_DeepConv = {100 * test_acc_resnet_deep_conv / len(test_dataset)} %")


#########
#########

# PART - B #

# Three level Resnet blocks with four fully-connected layers
class CNNResnet_DeepFC(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(CNNResnet_DeepFC, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 32)
        self.res3 = ResidualBlock(32, 32)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 1024)  # More neurons in the first fully connected layer
        self.fc2 = nn.Linear(1024, 512)  # Additional fully connected layer
        self.fc3 = nn.Linear(512, 256)  # Additional fully connected layer
        self.fc4 = nn.Linear(256, num_classes)  # Adjusted for the additional fully connected layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x



model_resnet_deep_fc = CNNResnet_DeepFC().to(device)

# # Printing model summary
# print(model_resnet_deep_fc)
# summary(model_resnet_deep_fc, (3, 32, 32))

# Defining the model hyperparameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_resnet_deep_fc.parameters(), lr=learning_rate)

# Training process begins
train_loss_list_resnet_deep_fc = []
train_accuracy_list_resnet_deep_fc = []
print(f"\n [EXPT - 4] : Training CNNResnet - Three level Resnet blocks with four fully-connected layers \n")
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}:', end=' ')
    train_loss = 0
    correct = 0
    total = 0

    # Iterating over the training dataset in batches
    model_resnet_deep_fc.train()
    for i, (images, labels) in enumerate(train_loader):
        # Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        # Calculating the model output and the cross entropy loss
        outputs = model_resnet_deep_fc(images)
        loss = criterion(outputs, labels)

        # Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_loss += loss.item()

    # Printing loss for each epoch
    train_loss_list_resnet_deep_fc.append(train_loss / len(train_loader))
    
    # Average accuracy for this epoch
    epoch_accuracy = 100 * correct / total
    train_accuracy_list_resnet_deep_fc.append(epoch_accuracy)

    print(f"Training loss = {train_loss_list_resnet_deep_fc[-1]}, Training Accuracy = {epoch_accuracy:.2f}%")



## Testing CNNResnet - Three level Resnet blocks with four fully-connected layers ###
test_acc_resnet_deep_fc =0
model_resnet_deep_fc.eval()

with torch.no_grad():
    #Iterating over the training dataset in batches
    for i, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        y_true = labels.to(device)

        #Calculating outputs for the batch being iterated
        outputs = model_resnet_deep_fc(images)

        #Calculated prediction labels from models
        _, y_pred = torch.max(outputs.data, 1)

        #Comparing predicted and true labels
        test_acc_resnet_deep_fc += (y_pred == y_true).sum().item()

    print(f"\n [EXPT - 4] : Test set accuracy with CNNResnet_DeepFC = {100 * test_acc_resnet_deep_fc / len(test_dataset)} %")



plt.figure(figsize=(14, 6))
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet, label='With CNN Resnet - Three level Resnet block with two fully-connected layers')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet_deep_conv, label='With CNN Resnet - Four level Resnet block with two fully-connected layer')
plt.plot(range(1, num_epochs + 1), train_accuracy_list_resnet_deep_fc, label='With CNN Resnet - Three level Resnet blocks with four fully-connected layers')
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")
plt.title("[EXPT - 4] : Training Accuracy Vs Epochs for CNN Resnet with different Network Depths")
plt.legend()
plt.show()



###### THE END #######