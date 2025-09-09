import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt                   # for plotting



########
########
########
########

### CASE - 1 : ###


# LOADING THE MNIST DATASET
training_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

testing_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# MAKING DATASETS [training_dataset, testing_dataset] ITERABLE
size_of_each_batch = 100
no_of_iters = 4000
num_epochs = int(no_of_iters / (len(training_dataset) / size_of_each_batch))

training_dataset_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=size_of_each_batch, shuffle=True)

testing_dataset_loader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=size_of_each_batch, shuffle=False)

# CREATE FeedForwardNeuralNetModel 
class FeedForwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.tan1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tan2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, output_dim)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.tan1(out)
        out = self.fc2(out)
        out = self.tan2(out)
        out = self.fc3(out)
        return out

# STEP 4: INSTANTIATE MODEL CLASS
input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedForwardNeuralNetModel(input_dim, hidden_dim, output_dim)

# STEP 5: INSTANTIATE LOSS CLASS
criterion = nn.CrossEntropyLoss()

# STEP 6: INSTANTIATE OPTIMIZER CLASS
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# STEP 7: TRAIN THE MODEL
l = []
it = []

itr = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(training_dataset_loader):
        images = images.view(-1, 28*28).requires_grad_()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        itr += 1
        
        if itr % 100 == 0:
            correct, total = 0, 0
            for images, labels in testing_dataset_loader:
                images = images.view(-1, 28*28).requires_grad_()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            it.append(itr)
            l.append(loss.item())
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(itr, loss.item(), accuracy))

plt.plot(it, l)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


