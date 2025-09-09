import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Neural_Net(nn.Module):
    def __init__(self, depth=2):
        super(Neural_Net, self).__init__()       
        self.input = nn.Linear(28*28, 50)       
        self.hidden = nn.ModuleList([nn.Linear(50, 50) for _ in range(depth)]) 
        self.output = nn.Linear(50, 10)          

    def forward(self, x):          
        x = torch.tanh(self.input(x)) 
        
        for layer in self.hidden: 
            x = torch.tanh(layer(x))     
        x = self.output(x)       

        return F.log_softmax(x, dim=1) 


transform_image = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) 

training_data = datasets.MNIST(root='.', train=True, download=True, transform=transform_image)  # fetch, download and load the MNIST training data

training_loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)       # create training data loader, fetches in batches of 64 samples at a time, shuffled at the beginning of each epoch

testing_data = datasets.MNIST(root='.', train=False, download=True, transform=transform_image)  # fetch, download and load the MNIST testing data

testing_loader = torch.utils.data.DataLoader(testing_data, batch_size=1000)                     # create training data loader, fetches in batches of 64 samples at a time


# function for training the Neural Network
def train(model, device, training_loader, optimizer, epoch):  
    model.train()                                         
    for batch_idx, (data, target) in enumerate(training_loader):  
        data, target = data.to(device), target.to(device)    
        data = data.view(data.size(0), -1)                   
        optimizer.zero_grad() 
        output = model(data)  
        loss = F.cross_entropy(output, target) 
        loss.backward() 
        optimizer.step() 
        # Print training progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} [{batch_idx*len(data)}/{len(training_loader.dataset)} ({100. * batch_idx / len(training_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# Test function
def test(model, device, testing_loader):  # Define testing procedure
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): 
        for data, target in testing_loader:  
            data, target = data.to(device), target.to(device)  
            data = data.view(data.size(0), -1)  
            output = model(data) 
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item() 
    test_loss /= len(testing_loader.dataset) 
    # Print test results
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testing_loader.dataset)} ({100. * correct / len(testing_loader.dataset):.0f}%)\n")

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use if GPU is available, otherwise use CPU
    given_depths = [2, 4, 8]                                                     # given depths of the hidden layers in neural network to experiment, as per CASE - 1

    for depth in given_depths:  
        print(f"\nTraining model with {depth} hidden layers : \n")
        model = Neural_Net(depth).to(device)                                     
        optimizer = optim.Adam(model.parameters())                        
                                                                          
        
        # performing Kaiming initialization
        for layer in model.modules():                 
            if isinstance(layer, nn.Linear):           
                nn.init.kaiming_normal_(layer.weight, nonlinearity='tanh') 
                
        for epoch in range(1, 11):                                   # train for 10 epochs
            train(model, device, training_loader, optimizer, epoch)  # training the model
            test(model, device, testing_loader)                      # testing the model after every epoch

if __name__ == '__main__': 
    main()  # run the main function
