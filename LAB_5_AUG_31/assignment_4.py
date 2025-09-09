import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv('.\car_evaluation.csv')

X = data
y = data['unacc']

X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=0.25, random_state=1, stratify=y_train_validation)

class TreeNode:
    def __init__(self,X,y,thresh,feature):
        self.X= X                         
        self.y= y                         
        self.thresh=thresh                
        self.feature=feature              
        self.children= []                 

    def add_child(self, child):         
        self.children.append(child)      

    def get_level(self):                 
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent

        return level

    def print_tree(self):                                      
        spaces = ' ' * self.get_level() * 3
        prefix = spaces + "|__" if self.parent else ""         
        print(prefix + "feature_idx: "+ str(self.feature))
        if self.children:
            for child in self.children:
                child.print_tree()                             
    
    # Check if the current node has a circular reference, i.e., the node or its children refer back to an ancestor node in their hierarchy.
    def has_circular_reference(self, visited=None):
        # Initialize visited set during the first call
        # This set keeps track of nodes we have already visited to detect cycles to prevent infinite loop
        if visited is None:  
            visited = set()

        if self in visited:
            print(f"Circular reference detected at node {self}")
            return True
        else:
            # If the node hasn't been visited before, add it to the visited set
            visited.add(self)

        has_circular_ref = False
        for child in self.children:
            # For each child of the current node, recursively call the function 
            # to check if the child has a circular reference
            if child.has_circular_reference(visited):           # recursive call for each child
                has_circular_ref = True
                break

        return has_circular_ref

    def check_parent_pointer(self):
        for child in self.children:
            if child.parent is not self:
                print(f"Parent pointer inconsistency detected at child {child} of node {self}")
                return True
            if child.check_parent_pointer():   # recursive call for each child
                return True

        return False


class Dec_Tree_Mod:
    def __init__(self, entropy_threshold=1.0):
        self.entropy_threshold = entropy_threshold
        self.root = None

    def fit(self, data, target):
        self.root = TreeNode(data, data.columns.tolist(), target, self.entropy_threshold)
        nodes = [self.root]
        self.print_tree(self.root)

        for feature in node.data.columns[:-1]:                                  
            gain = information_gain(node.y, groups)                        
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_groups = groups

        while nodes:
            
            current_node = nodes.pop()
            current_node.build_child()
            
            for child in current_node.children.values():
                if not child.decision:
                    nodes.append(child)



# EXPERIMENT - 1 :

thresholds = [0, 0.25, 0.5, 0.75, 1]
train_accuracies = []
valid_accuracies = []
tree_sizes = []


for threshold in thresholds:
   
    root = TreeNode(X_train, y_train, None, None)
    Dec_Tree_Mod(root, 10, threshold)
    
    t_accuracy = accuracy(y_train, predict(root, X_train))
    v_accuracy = accuracy(y_validation, predict(root, X_validation))
    
    t_accuracies.append(t_accuracy)
    v_accuracies.append(v_accuracy)
    
    tree_sizes.append(tree_size(root))


plt.plot(thresholds, train_accuracies, label='Training Accuracy ', marker='o')
plt.plot(thresholds, valid_accuracies, label='Validation Accuracy ', marker='o')
plt.xlabel('Threshold')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

plt.plot(thresholds, tree_sizes, label='Tree Size', marker='o')
plt.xlabel('Threshold')
plt.ylabel(' Size of Decision Tree')
plt.show()


best_threshold = thresholds[np.argmax(valid_accuracies)]
print(f"Best entropy threshold : {best_threshold}")



