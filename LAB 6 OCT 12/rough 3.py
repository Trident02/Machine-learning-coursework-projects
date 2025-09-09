# Import necessary libraries
import matplotlib.pyplot as plt              # Plotting library
import torch                                 # PyTorch library for tensors and neural networks operations
import torchvision.transforms as transforms  # Provides common image transformations
import torch.nn as nn                        # Neural network module of PyTorch
import torchvision.datasets as datasets      # Provides access to common datasets

########
########

# load the MNIST dataset for training
# the dataset is transformed to a tensor 
training_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

# load the MNIST dataset for testing 
testing_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# define parameters for making the datasets iterable in batches

size_of_each_batch = 100  # size of each batch
no_of_iters = 3600        # number of iterations
num_epochs = int(no_of_iters / (len(training_dataset) / size_of_each_batch))  # calculate number of epochs (3600/(60000/100) = 3600/600 = 6 epochs)

# convert training dataset into batches using DataLoader
training_dataset_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=size_of_each_batch, shuffle=True)

# convert testing dataset into batches (shuffle=False because the order is not important during testing)
testing_dataset_loader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=size_of_each_batch, shuffle=False)


########
########


# function to create a neural network layer (Linear transformation followed by Tanh activation)
def create_layer(input_dim, output_dim):
    layer = nn.Sequential(
        nn.Linear(input_dim, output_dim),  # linear transformation layer
        nn.Tanh()                          # Tanh activation function
    )
    nn.init.kaiming_normal_(layer[0].weight)  # initialize weights using Kaiming normalization

    return layer


########
########


# Define FeedForwardNeuralNetModel class to create the layers in the neural Network and perform Kaiming normalization
class FeedForwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(FeedForwardNeuralNetModel, self).__init__()
        
        layers = []                                         # empty list to hold layers
        layers.append(create_layer(input_dim, hidden_dim))  # add the first hidden layer
        
        # add subsequent hidden layers
        for i in range(n_layers - 1):
            layers.append(create_layer(hidden_dim, hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_dim))  # output layer without activation (it's applied implicitly in the CrossEntropyLoss)
        nn.init.kaiming_normal_(layers[-1].weight)        # initialize weights for the output layer using Kaiming normalization
        
        self.network = nn.Sequential(*layers)             # combine all layers into a sequential network

    # forward pass through the network
    def forward(self, x):
        return self.network(x)

########
########

# Reset the global variable to store each model's layer deviations separately
def reset_layer_deviation():
    global layer_deviation
    layer_deviation = {}

########
########

def evaluate_model(model, dataloader):
    """Evaluate the model on given dataloader and return accuracy"""
    correct, total = 0, 0                                          # initialize counters for the number of correct predictions and the total number of images
    with torch.no_grad():                                          # It speeds up the process since we don't need gradients during evaluation
        for images, labels in dataloader:                          # iterate over all batches of images and labels in the provided dataloader
            images = images.view(-1, 28*28)                        # reshape the images from their original shape (1x28x28) into a flat vector (784 = 28*28)
            outputs = model(images)                                # use the model to predict the labels for the current batch of images
            i, predicted = torch.max(outputs.data, 1)              # for each image in the batch, get the label (class) with the highest prediction score
            total += labels.size(0)                                # increase the total count by the number of images in the current batch
            correct += (predicted == labels).sum().item()          # Count how many of the predicted labels match the actual labels and increase the correct counter

    return 100 * correct / total                                   # return percentage accuracy

########
########

def perturb_and_evaluate(model, original_accuracy, n_layers):
    """Perturb each layer of the model and evaluate its performance"""
    global layer_deviation
    layer_count = 0

    # Perturb weights one-by-one 
    for idx, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name:
            # Create a copy of the original state_dict
            original_state_dict = model.state_dict()

            # Increment the layer_count
            layer_count += 1

            # Define layer naming based on the current layer_count and total number of layers
            if layer_count == 1:
                layer_name = 'Input_to_Hidden1'
            elif layer_count == n_layers + 1:
                layer_name = f'Hidden{n_layers}_to_Output'
            else:
                layer_name = f'Hidden{layer_count-1}_to_Hidden{layer_count}'

            # Perturb the current layer using Kaiming initialization
            noise = torch.randn_like(param) * (2 / (param.size(0)))**0.5
            param.data += noise                                            # adding noise to the parameters connecting the layers

            # Evaluate the model with the perturbed layer
            perturbed_accuracy = evaluate_model(model, testing_dataset_loader)
            deviation = original_accuracy - perturbed_accuracy             # calculating the percentage deviation of performance from the non-perturbed model

            # Store the deviation for this layer
            layer_deviation[layer_name] = deviation

            # Restore the original state_dict
            model.load_state_dict(original_state_dict)

########
########

# Function to train and test the model
def train_and_test_model(n_layers):
    # instantiate the neural network model with given number of layers
    model = FeedForwardNeuralNetModel(input_dim=28*28, hidden_dim=50, output_dim=10, n_layers=n_layers)

    criterion = nn.CrossEntropyLoss()  # using Cross entropy loss for classification
    learning_rate = 0.0001             # Learning rate for optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer is responsible for updating the model's weights during training

    loss_list = []        # List to store loss values
    iter = []             # List to store iteration numbers
    itr_count = 0         # Counter for iterations

    # Training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(training_dataset_loader):
            images = images.view(-1, 28*28).requires_grad_()         # Flatten the images
            optimizer.zero_grad()                                    # Zero out any accumulated gradients
            outputs = model(images)                                  # Forward pass
            loss = criterion(outputs, labels)                        # Compute loss according to the required criterion
            loss.backward()                                          # Backpropagation
            optimizer.step()                                         # Update weights using optimizer

            itr_count += 1                                           # Increase iteration counter

            # Every 100 iterations, evaluate the model on the test set
            if itr_count % 100 == 0:
                correct, total = 0, 0
                for images, labels in testing_dataset_loader:
                    images = images.view(-1, 28*28).requires_grad_() # reshape the images from their original shape (1x28x28) into a flat vector (784 = 28*28)
                    outputs = model(images)                          # use the model to predict the labels for the current batch of images
                    k, predicted = torch.max(outputs.data, 1)        # for each image in the batch, get the label (class) with the highest prediction score
                    total += labels.size(0)                          # increase the total count by the number of images in the current batch
                    correct += (predicted == labels).sum().item()    # Count how many of the predicted labels match the actual labels and increase the correct counter

                accuracy = 100 * correct / total  # compute accuracy
                iter.append(itr_count)            # store current iteration number
                loss_list.append(loss.item())     # store current loss value
                print(f'Hidden Layers: {n_layers}, Iteration: {itr_count}, Loss: {loss.item()}, Percentage Accuracy: {accuracy} %')

    # Plot loss values Vs. iterations
    plt.plot(iter, loss_list, label=f'{n_layers} layers')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')


    # Evaluate the fully trained model(i.e. after all the 3600 iterations) on test data
    original_accuracy = evaluate_model(model, testing_dataset_loader)
    print(f"\nAccuracy of the fully trained model (i.e. after 3600 iterations = 6 epochs), with {n_layers} hidden layers, on the test data : {original_accuracy:.2f}%\n")

    # Perturb each layer one-by-one keeping other layers intact and measure performance deviation from non-perturbed model
    perturb_and_evaluate(model, original_accuracy, n_layers)


########
########
########
########

### CASE - 1 : Compute accuracy on the MNIST test data for each of the model with different number of hidden layers ### 

print("MLFA Assignment 5 : \n")
print("CASE - 1 : \n")

all_layer_deviations = {}  # Using `all_layer_deviations` dictionary to store deviations for all model configurations,
                           # key is the number of hidden layers (n_layers) and the value is the layer deviations data obtained during this training and testing cycle

# Train and evaluate models with different depths
for n_layers in [2, 4, 8]:
    reset_layer_deviation()                                                  # Reset layer deviations for each model configuration
    print("\nTraining model with", n_layers, "number of hidden layers : \n")
    train_and_test_model(n_layers)                                           # training and testing the model with 2,4,8 hidden layers
    all_layer_deviations[n_layers] = layer_deviation.copy()                  # Store the layer deviations for the current model configuration (with n_layers hidden layers) 


plt.legend()  # show legend on the plot
plt.show()    # display the Plot of Loss Vs. Epochs(Iterations) for each model with different number of hidden layers


########
########
########
########

### CASE - 2 : Print the layers based on their performance deviation (in descending order) after perturbing the parameters ###

print("\nCASE - 2 : \n")

# After training all models, print the ranking of layers based on performance deviation for all the models
for n_layers, deviations in all_layer_deviations.items():

    print(f"\nRanking of layers based on Performance Deviation (in descending order) for {n_layers} hidden layers:\n")
    # sort the `deviations` dictionary in descending order based on the deviation values
    # this will rank layers based on their performance deviation when perturbed
    sorted_deviation = sorted(deviations.items(), key=lambda x: x[1], reverse=True) 

    # Enumerate through each sorted layer deviation entry
    # 'rank' is the ranking number (starting from 1), 'layer_name' is the name or description of the layer,
    # and 'deviation' is the performance deviation for that layer from the non-perturbed model
    for rank, (layer_name, deviation) in enumerate(sorted_deviation, 1):
        print(f"{rank}. When parameters connecting from {layer_name} is perturbed, - Performance Deviation from Non-Perturbed Model is : {deviation:.2f}%")