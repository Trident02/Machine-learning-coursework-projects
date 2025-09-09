import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

##################
##################
# \LAB 5 AUG 31\python rough_assignment_4.py > output.txt
# np.random.seed(42)

# Load the dataset
data = pd.read_csv('.\car_evaluation.csv')

# changing the names of the columns for convenient use
new_column_names = ['Price_Buying' , 'Price_Maintenance' , 'Doors','Persons','Lug_boot', 'Safety', 'Acceptability']
data.columns = new_column_names


# Manual Ordering of the data using Ordinal Encoder

buying_price_mapping = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
maintenance_price_mapping = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
doors_mapping = {'2': 0, '3': 1, '4': 2, '5more': 3}
persons_mapping = {'2': 0, '4': 1, 'more': 2}
lug_boot_mapping = {'small': 0, 'med': 1, 'big': 2}
safety_mapping = {'low': 0, 'med': 1, 'high': 2}
acceptability_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}

data['Price_Buying'] = data['Price_Buying'].map(buying_price_mapping)
data['Price_Maintenance'] = data['Price_Maintenance'].map(maintenance_price_mapping)
data['Doors'] = data['Doors'].map(doors_mapping)
data['Persons'] = data['Persons'].map(persons_mapping)
data['Lug_boot'] = data['Lug_boot'].map(lug_boot_mapping)
data['Safety'] = data['Safety'].map(safety_mapping)
data['Acceptability'] = data['Acceptability'].map(acceptability_mapping)



# display initial data insights
print(data)
print(data.shape)
print(data.describe())
print(data.info())                           # to get a view of how many columns and what data types

##################
##################

# SPLITTING THE DATA INTO 60:20:20 RATIO 

# splitting data into features (X) and target (y)
# X = data.drop(columns='Acceptability')
X = data
y = data['Acceptability']



# 80% for [training and validation], 20% for [testing]
# stratify = y makes sure that the train_valid and test datasets have the same proportion of class labels as the original dataset
X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 75% of 80% for training which is 60%, 25% of 80% for validation which is 20%
# stratify = y_train_val makes sure that the train and valid datasets have the same proportion of class labels as the original dataset
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=0.25, random_state=42, stratify=y_train_validation)

# display data insights after splitting
print(X)
print(y)
print(data.info()) 
print("Training dataset:", X_train.shape, y_train.shape)
print("Validation dataset:", X_validation.shape, y_validation.shape)
print("Testing dataset:", X_test.shape, y_test.shape)


##################
##################


# Function for calculating the entropy of a dataset
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)   # Unique values and their counts in the target column
    entropy = -np.sum([(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])   # From Shannon's Entropy Formula
    return entropy


def get_tree_size(node):

    if node is None:
        return 0
    # If the node has a decision (i.e., it's a leaf node), return 1.
    if node.decision is not None:
        return 1
     
    size = 1        # Start with 1 to count the current node.

    for child in node.children.values():   # Loop through all child nodes and recursively count their size
        size += get_tree_size(child)

    return size



def percentage_accuracy(model, X, y):
    return model.score(X, y) * 100


# Finding the best split for a node using the entropy threshold
def best_split(dataset, features, target, entropy_threshold, depth):
   
    if depth == 0 : 
        entropy_S = entropy(target)

    best_entropy = sys.maxsize                 # Initialize best_entropy to a high value; this value will be updated to find the feature with the lowest entropy
    best_feature = None                       # Initialize the best_feature to None, this will store the name of the feature that provides the best split

    for feature in features :
        if feature == "Acceptability":
            continue

        unique_values = np.unique(dataset[feature])      # get all unique values of the current feature; this is used to split the dataset on each unique value
        # For each unique value of the feature, filter the dataset where the feature equals that value
        # This gives us data subsets for each possible value of the feature
        print(unique_values)
        
        children_data = []
        for value in unique_values:
            subset = dataset[dataset[feature] == value]
            children_data.append(subset)

        # children_data = [dataset[dataset[feature] == value] for value in unique_values]  
        print(children_data)
        

        # Calculating the expected entropy for the current feature :
        # 1. Calculating the entropy for each data subset (child)
        # 2. Multiplying this entropy by the proportion of the total dataset that the child represents
        # 3. Summing up these expected entropies for all children
        expected_entropy = 0  # Initialize the expected entropy to zero

        for child in children_data:
            print("child : ", child)
            weight = len(child) / len(dataset)  # Calculate the weight of each child
            print("weight : ", weight)
            
            child_entropy = entropy(child['Acceptability'])  # Calculate the entropy of the target variable for the child
            weighted_entropy = weight * child_entropy  # Calculate the weighted entropy for the child
            expected_entropy += weighted_entropy  # Add the weighted entropy to the total expected entropy

        # expected_entropy = sum([(len(child) / len(dataset)) * entropy(child[target]) for child in children_data])
        
        if depth == 0 : 
            ig = entropy_S - expected_entropy
        else :
            ig = entropy(dataset['Acceptability']) - expected_entropy  # Information gain = Entropy of target - Expected Entropy of the features
        
        # if expected entropy of the feature is less than the current best_entropy
        # if expected entropy is less than a given threshold (this can prevent overfitting by not considering splits that result in very high certainty)
        if ig > 0 and expected_entropy < best_entropy and expected_entropy < entropy_threshold:
            best_entropy = expected_entropy
            best_feature = feature

    return best_feature



# A node for the decision tree
class TreeNode:

    def __init__(self, input_data, features, target, entropy_threshold, depth=0):
        self.input_data = input_data                                      # the dataset associated with this node
        self.features = features                              # all the feature column names that we can potentially split on
        self.target = target                                  #  name of the column that contains the output labels/classes i.e. 'Acceptability'
        self.entropy_threshold = entropy_threshold            # threshold of entropy we use to decide whether to split this node further or not
        self.depth = depth

        self.children = {}  # Dictionary to store child nodes based on feature values

        self.split_feature = None        # this property will store the feature on which this node will be split, It is initialized to None because we didn't decide on it yet

        # This property will store the final decision (class label) of the node if it's a leaf node
        # if the node can be split further, it will remain None
        self.decision = None

    # method to decide how to split the current node (if we split it) and then build its 
    # children nodes based on the unique values of the best feature for splitting
    def build_child(self):
        # best_split function identifies the best feature to split on based on 
        # the data of the current node, the available features, the target, and the entropy threshold
        self.split_feature = best_split(self.input_data, self.features, self.target, self.entropy_threshold, self.depth)

        # If there's no valid feature to split on (meaning this node is a leaf node),
        # assign the mode (most frequent class) of the target column in this node's data to the decision property
        if not self.split_feature:
            # print("Target:", self.target)
            self.decision = self.input_data['Acceptability'].mode()[0]
        else:
            # If there's a valid feature to split on, get all its unique values
            unique_values = np.unique(self.input_data[self.split_feature])

            for value in unique_values:  
                child_data = self.input_data[self.input_data[self.split_feature] == value]                         # Filter the data of the current node based on the feature's unique value
                child_node = TreeNode(child_data, self.features, self.target, self.entropy_threshold, depth=self.depth + 1)   # Create a new TreeNode (child node) using the filtered data               
                self.children[value] = child_node                                                      # Store this child node in the 'children' dictionary with the key being the unique value and the value being the child node itself
                
    # predicts the target value for a single data row using the decision tree
    def predict_single(self, row):
        if self.decision is not None:       # If the current node has a decision (it's a leaf node), return that decision
            return self.decision
        child_value = row[self.split_feature]  # If the current node is not a leaf node, get the value of the split feature from the input row
        
        if child_value in self.children:       # If this value has a corresponding child node in the 'children' dictionary:
            return self.children[child_value].predict_single(row)   # Recursively predict using the appropriate child node and the same input row
        else:
            # If the value from the input row doesn't match any of the children nodes (wasn't seen in training data),
            # return the most frequent class in the current node as a fall-back decision.
            
            return self.input_data[self.target].mode()[0]
        

    def __str__(self):
        entropy_str = f", Entropy: {self.entropy}" if hasattr(self, 'entropy') else ""
        return f"Feature: {self.split_feature}{entropy_str}, Depth: {self.depth}"

# Dec_Tree_Mod is a modified decision tree that uses an entropy threshold for making splits
class Dec_Tree_Mod:
    # Constructor for Dec_Tree_Mod class
    def __init__(self, entropy_threshold=1.0):
        self.entropy_threshold = entropy_threshold   # nodes with entropy above this threshold will be split
        self.root = None

    # fit the decision tree model to the given data
    def fit(self, input_data, target):
        # root node of the tree using the input data, its features, the target variable, and the entropy threshold
        self.root = TreeNode(input_data, input_data.columns.tolist(), target, self.entropy_threshold, 0)
        nodes = [self.root]                           # Initialize a list (stack) with the root node
        self.print_tree(self.root)
        
        while nodes:
            
            current_node = nodes.pop()           # Pop from the back for depth-first traversal
            current_node.build_child()           # Build child nodes for the current node
            # If the current node has children and they do not have a decision (i.e., they're not leaf nodes), add them to the stack
            for child in current_node.children.values():
                if not child.decision:
                    nodes.append(child)

    # Method to predict the class labels for the given input data X
    def predict(self, X):
        return X.apply(self.root.predict_single, axis=1)   # Apply the predict_single method of the root node to each row of the input data X

    # Method to evaluate the accuracy of the model's predictions
    def score(self, X, y): 
        predictions = self.predict(X)                   # Predict the class labels for the input data X
        return (predictions == y).mean()
    
    def print_tree(self, node, prefix=""):
        # Base case: if node is None, just return
        if not node:
            return
        
        # Print current node's information
        print(prefix + str(node))
        
        # Recursive call for each child
        for i, child in enumerate(node.children):
            child_prefix = prefix + "--> Child " + str(i+1) + " of " + str(node) + ": "
            self.print_tree(child, child_prefix)




##################
##################
##################
##################
##################



# EXPERIMENT - 1 : 


thresholds = [0, 0.25, 0.5, 0.75, 1]
train_accuracies = []
validation_accuracies = []
tree_sizes = []

for threshold in thresholds:
    tree = Dec_Tree_Mod(entropy_threshold=threshold)
    #print(tree.entropy_threshold)
    tree.fit(X_train, y_train)
    train_accuracies.append(percentage_accuracy(tree, X_train, y_train))
    validation_accuracies.append(percentage_accuracy(tree, X_validation, y_validation))
    tree_sizes.append(get_tree_size(tree.root))



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, train_accuracies, label='Training Accuracy', marker='o')
plt.plot(thresholds, validation_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Entropy Threshold')
plt.ylabel('Percentage Accuracy')
plt.title('Accuracy vs. Entropy Threshold')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(thresholds, tree_sizes, marker='o', color='green')
plt.xlabel('Entropy Threshold')
plt.ylabel('Size of Decision Tree')
plt.title('Tree Size vs. Entropy Threshold')

plt.tight_layout()
plt.show()

best_threshold = thresholds[np.argmax(validation_accuracies)]
print(f"Best entropy threshold based on validation accuracy: {best_threshold}")
