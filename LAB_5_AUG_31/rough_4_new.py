# NAME : Shiva Ganesh Reddy Lakkasani
# ROLL NUMBER : 20EE10069
# MLFA (AI42001) Assignment - 4



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



##################
##################

# np.random.seed(1)

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

# Define reverse mappings for printing the Rules for classification in EXPERIMENT - 3
reverse_buying_price_mapping = {v: k for k, v in buying_price_mapping.items()}
reverse_maintenance_price_mapping = {v: k for k, v in maintenance_price_mapping.items()}
reverse_doors_mapping = {v: k for k, v in doors_mapping.items()}
reverse_persons_mapping = {v: k for k, v in persons_mapping.items()}
reverse_lug_boot_mapping = {v: k for k, v in lug_boot_mapping.items()}
reverse_safety_mapping = {v: k for k, v in safety_mapping.items()}
reverse_acceptability_mapping = {v: k for k, v in acceptability_mapping.items()}

reverse_mappings = {
    'Price_Buying': reverse_buying_price_mapping,
    'Price_Maintenance': reverse_maintenance_price_mapping,
    'Doors': reverse_doors_mapping,
    'Persons': reverse_persons_mapping,
    'Lug_boot': reverse_lug_boot_mapping,
    'Safety': reverse_safety_mapping,
    'Acceptability': reverse_acceptability_mapping
}

# display initial data insights
print(data)
print(data.shape)
print(data.describe())
print(data.info())                           # to get a view of how many columns and what data types

##################
##################

# SPLITTING THE DATA INTO 60:20:20 RATIO 

# splitting data into features (X) and target (y)
X = data.drop('Acceptability', axis=1)
y = data['Acceptability']



# 80% for [training and validation], 20% for [testing]
# stratify = y makes sure that the train_valid and test datasets have the same proportion of class labels as the original dataset
X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

# 75% of 80% for training which is 60%, 25% of 80% for validation which is 20%
# stratify = y_train_validation makes sure that the train and valid datasets have the same proportion of class labels as the original dataset
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=0.25, random_state=1, stratify=y_train_validation)

# display data insights after splitting
print(X)
print(y)
print(data.info()) 
print("Training dataset:", X_train.shape, y_train.shape)
print("Validation dataset:", X_validation.shape, y_validation.shape)
print("Testing dataset:", X_test.shape, y_test.shape)


##################
##################

# Compute the entropy for a set of labels
def entropy(y):
    unique, counts = np.unique(y, return_counts=True)         # get the unique values in the target label
    total = counts.sum()
    calculated_entropy = 0

    for count in counts:
        probability = count / total
        if probability > 0:
            calculated_entropy += probability * np.log2(probability)

    return -calculated_entropy



# Compute the information gain of a split
def information_gain(parent_y, groups):
    parent_entropy = entropy(parent_y)
    total_samples = len(parent_y)
    weighted_child_entropy = sum([(len(group) / total_samples) * entropy(group) for group in groups])
    return parent_entropy - weighted_child_entropy

# defining a class 
class TreeNode:
    def __init__(self,X,y,thresh,feature):
        self.X= X                         # storing the input data at a node
        self.y= y                         # storing the class labels
        self.thresh=thresh                # storing the threshold for splitting
        self.feature=feature              # storing the feature used for splitting
        self.children= []                 # list of children nodes
        self.parent= None                 # storing the parent node
        self.entropy = entropy(y)         # Add this line to store entropy for this node
        self.node_value = []

    def add_child(self, child):           # adding a child node
        child.parent = self               # setting the node as child's parent
        self.children.append(child)       # storing the child node in the list of children nodes

    def get_level(self):                  # get the depth of a node
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent

        return level

    def print_tree(self):                                      # displaying the tree formed hierarchically
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""         # using indentation and |_ as representation of depth & parent-child relation
        print(prefix + "feature_idx: "+ str(self.feature))
        if self.children:
            for child in self.children:
                child.print_tree()                             # recursively use the method for displaying the subtrees
    
    
    # Modify the get_rules method
    def get_rules(self, path=[]):             # Recursive function to generate rules from the tree
        if not self.children:                 # Leaf node
            rules = []
            for feature, threshold in path:
                cat_value = reverse_mappings[feature][threshold]
                rules.append(f"{feature} = {cat_value}")
            antecedent = " AND ".join(rules)
            consequent = reverse_mappings['Acceptability'][self.y.value_counts().idxmax()]
            return [f"IF {antecedent} THEN {consequent}"]

        # If not leaf node, traverse its children
        rules_list = []
        for child in self.children:
            rules_list.extend(child.get_rules(path + [(self.feature, child.node_value)]))

        return rules_list

    # Print the rules method remains unchanged
    def print_rules(self, i):
        rules_list = self.get_rules()
        for rule in rules_list:
            print(rule)
        print("Total Number of Rules for Classification in Experiment - ",i ," : ",len(rules_list))
        

  
# Implementing Dec_Tree_Mod for building a Decision Tree using the entropy and Information Gian for each feature
def Dec_Tree_Mod(node, depth, entropy_threshold):
    if depth > 0 and len(node.X) >= 1 and node.entropy > entropy_threshold:  # Check depth, data samples, and entropy
        best_feature = None
        best_gain = -float('inf')

        for feature in node.X.columns:  
            # Group by feature values and get associated 'Acceptability' values
            grouped_data = pd.concat([node.X, node.y], axis=1).groupby(feature)
            groups = [grp for _, grp in grouped_data]
                
            target_groups = [grp['Acceptability'].values for grp in groups]
            
            gain = information_gain(node.y, target_groups)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_feature is not None:                                   # Checks if a feature for splitting was identified 
            node.feature = best_feature                                # Store the best feature used for splitting
            unique_values = node.X[best_feature].unique()

            for value in unique_values:
                filtered_data = node.X[node.X[best_feature] == value]  # Filter X based on best feature's value
                child_df = filtered_data.drop(columns=[best_feature])  # Drop the best feature column
                child_y = node.y[filtered_data.index]                  # Filter y based on indices of filtered X
                child = TreeNode(child_df, child_y, entropy_threshold, best_feature) # Creates a new tree node with this subset
                child.node_value = value
                node.add_child(child)                                  # Adds the new node as a child of the current node
                Dec_Tree_Mod(child, depth-1, entropy_threshold)        # Recursively calls Dec_Tree_Mod on the child node, decreasing the depth by 1
    else: 
        return


def tree_size(node):         # Recursively calculates the number of nodes in the tree
    if not node.children:
        return 1
    return 1 + sum(tree_size(child) for child in node.children)


# Predict class for a single sample
def predict_single(node, x):                    
    if not node.children:                       # If we are at a leaf node i.e. no child node
        return node.y.value_counts().idxmax()   # Return the most common class at this node 

    
    for child in node.children:                 # If not a leaf node, find the right child node to move
        if child.node_value == x[node.feature]:
            return predict_single(child, x)

    # If none of the child thresholds match :
    # return the most common class at this node
    return node.y.value_counts().idxmax()


def predict(node, X):
    return X.apply(lambda x: predict_single(node, x), axis=1)    # Applying the predict_single function for each row in the dataset


# Computing Accuracy in percentage (%)
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean() * 100



##################
##################
##################
##################
##################


# EXPERIMENT - 1 :


# Lists to store results
thresholds = [0, 0.25, 0.5, 0.75, 1]
train_accuracies = []
valid_accuracies = []
tree_sizes = []


for threshold in thresholds:
    # Build tree with current threshold
    root = TreeNode(X_train, y_train, None, None)
    Dec_Tree_Mod(root, 1000, threshold)
    
    # Evaluate on training and validation data 
    train_accuracy = accuracy(y_train, predict(root, X_train))
    valid_accuracy = accuracy(y_validation, predict(root, X_validation))
    
    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)
    
    # Record tree size
    tree_sizes.append(tree_size(root))


# Plotting  Percentage Accuracy vs threshold hyperparameter on training and validation data
plt.figure(figsize=(13,8))
plt.plot(thresholds, train_accuracies, label='Training Accuracy (in %)', marker='o')
plt.plot(thresholds, valid_accuracies, label='Validation Accuracy (in %)', marker='o')
plt.xlabel('Entropy Threshold')
plt.ylabel('Percentage Accuracy (%)')
plt.title('[EXPT - 1] PART - A : Percentage Accuracy (in %) Vs Entropy Threshold Hyperparameter')
plt.legend()
plt.show()

# Plotting Percentage Accuracy vs threshold hyperparameter on training and validation data using BAR CHARTS
bar_width = 0.28
index = np.arange(len(thresholds))

plt.figure(figsize=(13,8))
bar1 = plt.bar(index, train_accuracies, bar_width, label='Training Accuracy', alpha=0.7)
bar2 = plt.bar(index + bar_width, valid_accuracies, bar_width, label='Validation Accuracy', alpha=0.7)

plt.xlabel('Entropy Threshold')
plt.ylabel('Percentage Accuracy (%)')
plt.title('[EXPT - 1] PART - A : Percentage Accuracy (in %) Vs Entropy Threshold Hyperparameter using Bar Charts')
plt.xticks(index + bar_width/2, thresholds)  # Positioning the x-labels in the center of the grouped bars
plt.legend()
plt.tight_layout()
plt.show()


# Plotting sizes of Decision Trees vs Entropy Threshold Hyperparameter
plt.figure(figsize=(13,8))
plt.plot(thresholds, tree_sizes, label='Tree Size', marker='o')
plt.xlabel('Entropy Threshold')
plt.ylabel('Number of Nodes')
plt.title('[EXPT - 1] PART - B :  Size of Decision Tree [Number of Nodes in the Decision Tree] vs Entropy Threshold Hyperparameter')
plt.show()


best_threshold_validation = thresholds[np.argmax(valid_accuracies)]
print(f"[EXPT - 1] Best entropy threshold based on Validation accuracy : {best_threshold_validation}")


##################
##################
##################
##################
##################


# EXPERIMENT - 2 :

##################
##################

# PART - A : 
root_best_threshold = TreeNode(X_train, y_train, None, None)
Dec_Tree_Mod(root_best_threshold, 10, best_threshold_validation)

train_accuracy_best = accuracy(y_train, predict(root_best_threshold, X_train))
test_accuracy_best = accuracy(y_test, predict(root_best_threshold, X_test))

print(f"[EXPT - 2] PART - A : Training Accuracy with best threshold: {train_accuracy_best:.2f}%")
print(f"[EXPT - 2] PART - A : Testing Accuracy with best threshold: {test_accuracy_best:.2f}%")
print(f"Please wait for 3-4 mins the plots for [EXPT - 2] PART - B are BEING COMPUTED !")

##################
##################

# PART - B : 

training_accuracies_after_each_branch, validation_accuracies_after_each_branch = [], []

# Implementing build_tree_stepwise() method to keep track on percentage accuracy after every branch formation
def Dec_Tree_Mod_stepwise(node, depth, entropy_threshold, root_node):
    print(f"Please WAIT ! The Plots for EXPT - 2,PART - B are being computed !")
    if depth > 0 and len(node.X) >= 1 and node.entropy > entropy_threshold:  # Check depth, data samples, and entropy
        best_feature = None
        best_gain = -float('inf')

        for feature in node.X.columns: 
            # Group by feature values and get associated 'Acceptability' values
            grouped_data = pd.concat([node.X, node.y], axis=1).groupby(feature)
            groups = [grp for _, grp in grouped_data]
                
            target_groups = [grp['Acceptability'].values for grp in groups]
            
            gain = information_gain(node.y, target_groups)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_feature is not None:
            node.feature = best_feature                               # Store the best feature used for splitting
            unique_values = node.X[best_feature].unique()

            for value in unique_values:
                filtered_data = node.X[node.X[best_feature] == value]    # Filter X based on best feature's value
                child_df = filtered_data.drop(columns=[best_feature])    # Drop the best feature column
                child_y = node.y[filtered_data.index]                    # Filter y based on indices of filtered X
                child = TreeNode(child_df, child_y, entropy_threshold, best_feature) # Creates a new tree node with this subset
                child.node_value = value
                node.add_child(child)                                    # Adds the new node as a child of the current node                

                # Calculate accuracies and store them
                training_accuracies_after_each_branch.append(accuracy(y_train, predict(root_node, X_train)))
                validation_accuracies_after_each_branch.append(accuracy(y_validation, predict(root_node, X_validation)))

                Dec_Tree_Mod_stepwise(child, depth-1, entropy_threshold, root_node)  # Recursively calls Dec_Tree_Mod_stepwise on the child node, decreasing the depth by 1
    else: 
        return
    
# Build the tree stepwise
root_stepwise = TreeNode(X_train, y_train, None, None)
Dec_Tree_Mod_stepwise(root_stepwise, 10, best_threshold_validation, root_stepwise )


print(f"[EXPT - 2] PART - B : Training Accuracy after each branch: ")
print(training_accuracies_after_each_branch)
# Plotting training accuracy after each branch formation
plt.figure(figsize=(13,8))
plt.plot(range(len(training_accuracies_after_each_branch)), training_accuracies_after_each_branch, label='Train Accuracy', marker='o')
plt.xlabel('Number of Branches')
plt.ylabel('Training Accuracy (%)')
plt.title('[EXPT - 2] PART - B : Percentage Accuracy on Training Dataset after Every Branch Formation')
plt.legend()
plt.show()


print(f"[EXPT - 2] PART - B : Validation Accuracy after each branch: ")
print(validation_accuracies_after_each_branch)
# Plotting validation accuracy after each branch formation
plt.figure(figsize=(13,8))
plt.plot(range(len(validation_accuracies_after_each_branch)), validation_accuracies_after_each_branch, label='Validation Accuracy', marker='o')
plt.xlabel('Number of Branches')
plt.ylabel('Validation Accuracy (%)')
plt.title('[EXPT - 2] PART - B : Percentage Accuracy on Validation Dataset after Every Branch Formation')
plt.legend()
plt.show()


##################
##################


# PART - C : 

# Implement early stopping when validation accuracy starts decreasing and compare the accuracies

def Dec_Tree_Mod_early_stopping(node, depth, entropy_threshold, root_node, prev_val_accuracy=0.0):
    if depth > 0 and len(node.X) >= 1 and node.entropy > entropy_threshold:  # Check depth, data samples, and entropy
        best_feature = None
        best_gain = -float('inf')

        for feature in node.X.columns:  
            # Group by feature values and get associated 'Acceptability' values
            grouped_data = pd.concat([node.X, node.y], axis=1).groupby(feature)
            groups = [grp for _, grp in grouped_data]
                
            target_groups = [grp['Acceptability'].values for grp in groups]
            
            gain = information_gain(node.y, target_groups)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_feature is not None:
            node.feature = best_feature                                 # Store the best feature used for splitting
            unique_values = node.X[best_feature].unique()
            for value in unique_values:
                filtered_data = node.X[node.X[best_feature] == value]    # Filter X based on best feature's value
                child_df = filtered_data.drop(columns=[best_feature])    # Drop the best feature column
                child_y = node.y[filtered_data.index]                    # Filter y based on indices of filtered X
                child = TreeNode(child_df, child_y, entropy_threshold, best_feature) # Creates a new tree node with this subset
                child.node_value = value
                node.add_child(child)                                    # Adds the new node as a child of the current node

                val_accuracy = accuracy(y_validation, predict(root_node, X_validation))
                
                if val_accuracy < prev_val_accuracy: # Stop early if validation accuracy decreases
                    return
                
                prev_val_accuracy = val_accuracy
                Dec_Tree_Mod_early_stopping(child, depth-1, entropy_threshold,root_node, prev_val_accuracy) # Recursively calls Dec_Tree_Mod_early_stopping on the child node, decreasing the depth by 1
    else:
        return


# Build the tree with early stopping
root_early_stopping = TreeNode(X_train, y_train, None, None)
Dec_Tree_Mod_early_stopping(root_early_stopping, 10, best_threshold_validation, root_early_stopping)

train_accuracy_early = accuracy(y_train, predict(root_early_stopping, X_train))
test_accuracy_early = accuracy(y_test, predict(root_early_stopping, X_test))

print(f"[EXPT - 2] PART - C : Training Accuracy with early stopping: {train_accuracy_early:.2f}%")
print(f"[EXPT - 2] PART - C : Testing Accuracy with early stopping: {test_accuracy_early:.2f}%")

##################
##################

# Implementing to find out the total number of nodes which correspond to the case when validation percentage starts to decrease

decrease_start_index = None
for i in range(1, len(validation_accuracies_after_each_branch)):
    if validation_accuracies_after_each_branch[i] < validation_accuracies_after_each_branch[i-1]:
        decrease_start_index = i
        break

# calculating the number of nodes in the tree up to that branch, when the validation accuracy starts decreasing
def compute_nodes_until(node, depth, until_depth):
    if depth == until_depth or node is None:
        return 0
    count = 1
    for child in node.children:
        count += compute_nodes_until(child, depth+1, until_depth)
    return count

if decrease_start_index is not None:
    nodes_until_decrease = compute_nodes_until(root_stepwise, 0, decrease_start_index)
    print(f"[EXPT - 2] PART - C : Number of nodes when validation percentage accuracy starts to decrease: {nodes_until_decrease}")
else:
    print(f"[EXPT - 2] PART - C : Validation percentage did not decrease within the given depth of the tree.")



##################
##################
##################
##################
##################


# EXPERIMENT - 3 :


# print the rules for classification in a readable format ANTECEDENT (IF) => CONSEQUENT (THEN)
def extract_rules(node, depth=0, max_depth=10000):
    # Stop recursion if depth exceeds max_depth
    if depth > max_depth:
        print(f"Reached maximum recursion depth at depth {depth}")
        return
    
    if not node.children:
        # at leaf node, get the most common class
        consequent = node.y.value_counts().idxmax()
        antecedents = []

        # traverse upwards to the root node to collect conditions
        while node.parent is not None:
            antecedents.append((node.parent.feature, node.thresh))
            node = node.parent

        # reverse the list so rules are in the order from root to leaf
        antecedents = antecedents[::-1]

        rule = " AND ".join([f"{ant[0]} == {ant[1]}" for ant in antecedents])
        print(f"IF {rule} THEN Class = {consequent}")

    for child in node.children:
        print(f"Processing node {node} at depth {depth}")  # Diagnostic information
        extract_rules(child, depth+1, max_depth)


print("Rules for classification for the Decision Tree in Experiment 1 :")
root_best_threshold.print_rules(i=1)


print("\nRules for classification for the Decision Tree in Experiment 2 :")
root_early_stopping.print_rules(i=2)


########### THE END ###############

