import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

#Defining plotting settings
plt.rcParams['figure.figsize'] = 14, 6

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

#Initializing normalizing transform for the dataset
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = mean,
                                     std = std)])

#Downloading the CIFAR10 dataset into train and test sets
train_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/train", train=True,
    transform=normalize_transform,
    download=True)



test_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR10/test", train=False,
    transform=normalize_transform,
    download=True)

#Generating data loaders from the corresponding datasets
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)  # MMMR --- Shuffle needs to be true.
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # --- Conv layer Set 1
        self.l1_conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.l1_relu = torch.nn.ReLU()
        self.l1_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        # --- Conv layer Set 2
        self.l2_conv = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.l2_relu = torch.nn.ReLU()
        self.l2_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        # --- Conv layer Set 3
        self.l3_conv = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.l3_relu = torch.nn.ReLU()
        self.l3_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        # --- Conv layer Set 4
        self.l4_conv = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.l4_relu = torch.nn.ReLU()
        self.l4_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        # --- Conv layer Set 5
        self.l5_conv = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.l5_relu = torch.nn.ReLU()
        self.l5_maxpool = torch.nn.MaxPool2d(kernel_size=2)
        # --- Conv layer Set 6
        self.l6_conv = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.l6_relu = torch.nn.ReLU()
        # No maxpooling for 6th layer to preserve spatial dimensions
        # --- Conv layer Set 7
        self.l7_conv = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.l7_relu = torch.nn.ReLU()
        # No maxpooling for 7th layer to preserve spatial dimensions
        
        # Flatten layer for fully connected layers
        self.fc_flat = torch.nn.Flatten()
        # Adjust the input features of fc_lin1 according to the output of the last conv layer
        self.fc_lin1 = torch.nn.Linear(512*2*2, 1024)  # The dimension here depends on the input image size and the architecture
        self.fc_relu = torch.nn.ReLU()
        self.fc_lin2 = torch.nn.Linear(1024, 10)
        
    def forward(self, x):
        # --- Conv layer Set 1
        x = self.l1_conv(x)
        x = self.l1_relu(x)
        x = self.l1_maxpool(x)
        # ---- Conv layer Set 2
        x = self.l2_conv(x)
        x = self.l2_relu(x)
        x = self.l2_maxpool(x)
        # ---- Conv layer Set 3
        x = self.l3_conv(x)
        x = self.l3_relu(x)
        x = self.l3_maxpool(x)
        # ---- Conv layer Set 4
        x = self.l4_conv(x)
        x = self.l4_relu(x)
        x = self.l4_maxpool(x)
        # ---- Conv layer Set 5
        x = self.l5_conv(x)
        x = self.l5_relu(x)
        x = self.l5_maxpool(x)
        # ---- Conv layer Set 6
        x = self.l6_conv(x)
        x = self.l6_relu(x)
        # ---- Conv layer Set 7
        x = self.l7_conv(x)
        x = self.l7_relu(x)
        # FC layers
        x = self.fc_flat(x)
        x = self.fc_lin1(x)
        x = self.fc_relu(x)
        x = self.fc_lin2(x)
        return x

# Printing model summary


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_to_analyse = CNN().to(device)
# print(model_to_analyse)

# summary(model_to_analyse, (3, 32, 32))


#Selecting the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN().to(device)

#Defining the model hyper parameters
num_epochs = 50
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training process begins
train_accuracy_list = []
train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end = ' ')
    train_loss = 0
    correct = 0
    total = 0
    

    #Iterating over the training dataset in batches
    model.train()
    for i, (images, labels) in enumerate(train_loader):

        #Extracting images and target labels for the batch being iterated
        images = images.to(device)
        labels = labels.to(device)

        #Calculating the model output and the cross entropy loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Updating weights according to calculated loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
        train_loss += loss.item()

    #Printing loss for each epoch
    train_loss_list.append(train_loss/len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")

    epoch_accuracy = 100 * correct / total
    train_accuracy_list.append(epoch_accuracy)

#Plotting loss for all epochs
plt.plot(range(1,num_epochs+1), train_accuracy_list)
plt.xlabel("Number of epochs")
plt.ylabel("Training Accuracy")
