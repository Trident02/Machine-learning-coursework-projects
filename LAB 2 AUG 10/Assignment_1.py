# NAME : Shiva Ganesh Reddy Lakkasani
# ROLL NUMBER : 20EE10069
# MLFA (AI42001) Assignment - 1




# importing essential libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# importing dataset - 1
input_dataset_1 = np.load('Dataset-1\inputs_Dataset-1.npy')
output_dataset_1 = np.load('Dataset-1\outputs_Dataset-1.npy')

# correcting output dataset - 1 as suggested by Professor (i.e. changing all zeroes to -1's)
output_dataset_1[output_dataset_1 == 0] = -1         # Applying Boolean Mask

# Implementation of PLA for DataSet - 1 :

X_1 = input_dataset_1    # Given input dataset 1
y_1 = output_dataset_1   # Given output dataset 1

# defining a class called "Perceptron"
class Perceptron:
    def __init__(self, MAX_iters) :
        self.MAX_iters = MAX_iters
        
    def predict(self, inputs):
        return 1 if np.dot(inputs, self.weights) >= 0 else -1
    
    def plot_misclassification_history(self, exp_number):
        plt.figure(figsize=(10,6))
        plt.plot(self.misclassified_history)
        plt.xlabel('Iteration')
        plt.ylabel('Number of Misclassified Instances')
        plt.title(f'For Experiment : {exp_number}, Iteration Vs. Number of Misclassified Instances')
        plt.grid(True)
        plt.show()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # List to keep track of misclassified counts at each iteration
        self.misclassified_history = []
        #initialize the parameters
        self.weights = np.random.uniform(-5.0, 5.0, n_features)  # Here, n_features includes the extra dimension for '1', hence, we are not doing, 'n_features+1'

        for _ in range(self.MAX_iters):
            num_of_misclassified = 0
            for inputs, y_label in zip(X, y):
                y_prediction = self.predict(inputs)
                if y_prediction != y_label:
                    self.weights += y_label * inputs
                    num_of_misclassified += 1
            # Store the number of misclassified for this iteration
            self.misclassified_history.append(num_of_misclassified)


# Training PLA for Dataset - 1
perceptron_1a = Perceptron(2000)                    # passing the parameter 'MAX_iters' to the object 'perceptron'
perceptron_1a.fit(X_1, y_1)                         # calling the fit method in 'perceptron' class


###################
###################
###################
###################
###################
###################


# EXPERIMENT - 1 :

# EXPT - 1, PART - A and B : [ calculating the values of the performance metrics for different values of 'K' &&
# Report the mean and variances of the performance metrics computed over all the folds for each 'K' ]



for i in range(10):
    k = i + 2
    kf = KFold(n_splits=k)

    knn_kfold = []
    prediction_kfold = []
    metrics_kfold = []

    for train_index, test_index in kf.split(X_1):      # this loop will run for 'k' times
        X_train, X_test = X_1[train_index], X_1[test_index]
        y_train, y_test = y_1[train_index], y_1[test_index]

        knn_kfold.append(KNeighborsClassifier(n_neighbors=5))
        knn_kfold[-1].fit(X_train, y_train)
        prediction = knn_kfold[-1].predict(X_test)

        acc = accuracy_score(y_test, prediction)     # TP+TN / (TP + FP + TN + FN)
        prec = precision_score(y_test, prediction)   # TP / (TP+FP)
        rec = recall_score(y_test, prediction)       # TP / (TP+FN)
        f1 = f1_score(y_test, prediction)            # 2*TP / (2*TP + FP + FN)

        prediction_kfold.append(prediction)
        metrics_kfold.append([acc, prec, rec, f1])
        
    print("Accuracy Score, Precision Score, Recall Score and F1 Score for K-Fold where K = ",k,"are : \n")
    
    for metrics in metrics_kfold:
        print(metrics)

    #find mean and variance for performance metrics of the models
    metrics_array = np.array(metrics_kfold)
    mean_metrics = np.mean(metrics_array, axis=0)
    var_metrics = np.var(metrics_array, axis=0)

    print("\nMean of the metrics for K = ",k,"are : \n", mean_metrics)
    print("\nVariance of the metrics for K = ",k,"are : \n", var_metrics)
    print("\n")

# EXPT - 1, PART - C : [80:20 split and plotting iterations Vs. number of misclassified instances]

from sklearn.model_selection import train_test_split     
X_train_1c, X_test_1c = train_test_split(X_1, test_size=0.2, shuffle=False)   # Making an 80:20 train-test split of dataset X_1
y_train_1c, y_test_1c = train_test_split(y_1, test_size=0.2, shuffle=False)   # Making an 80:20 train-test split of dataset y_1

perceptron_1c = Perceptron(2000)                    # passing the parameter 'MAX_iters' to the object 'perceptron'
perceptron_1c.fit(X_train_1c, y_train_1c)           # calling the fit method in 'perceptron' class
perceptron_1c.plot_misclassification_history(1)     # plot the iteration Vs. number of misclassification counts


###################
###################
###################
###################
###################
###################


# EXPERIMENT - 2 :

# importing dataset - 2
input_dataset_2 = np.load('Dataset-2\inputs_Dataset-2.npy')
output_dataset_2 = np.load('Dataset-2\outputs_Dataset-2.npy')

# correcting output dataset - 2 as suggested by Professor (i.e. changing all zeroes to -1's)
output_dataset_2[output_dataset_2 == 0] = -1         # Applying Boolean Mask

X_2 = input_dataset_2    # Given input dataset 2
y_2 = output_dataset_2   # Given output dataset 2


# EXPT - 2, PART - A :

# Making an 80:20 train-test split
from sklearn.model_selection import train_test_split
X_train_2a, X_test_2a = train_test_split(X_2, test_size=0.2, shuffle=False)
y_train_2a, y_test_2a = train_test_split(y_2, test_size=0.2, shuffle=False)

perceptron_2a = Perceptron(2000)                    # Implementation of PLA for DataSet - 2 and passing the parameter 'MAX_iters' to the object 'perceptron'
perceptron_2a.fit(X_train_2a, y_train_2a)           # calling the fit method in 'perceptron' class to train PLA
perceptron_2a.plot_misclassification_history(2)     # plot the iteration Vs. number of misclassification counts

# EXPT - 2, PART - B :

# Included in the PDF Report.


# EXPT - 2, PART - C : [ calculating the values of the performance metrics for the test data ]


# Predict on train set
size_train_2a = y_train_2a.shape
y_pred_train_2a = np.zeros(size_train_2a)
i = 0
for x_i in X_train_2a:
    y_pred_train_2a[i] = perceptron_2a.predict(x_i)
    i += 1

# Calculating accuracy for train data
accuracy_train_2a = accuracy_score(y_train_2a, y_pred_train_2a)

print(f"\nCalculating accuracy for train data in EXPERIMENT - 2 :")
print(f"\nAccuracy for train data : {accuracy_train_2a}")                           # TP+TN / (TP + FP + TN + FN)


# Predict on test set
size_test_2a = y_test_2a.shape
y_pred_test_2a = np.zeros(size_test_2a)
i = 0
for x_i in X_test_2a:
    y_pred_test_2a[i] = perceptron_2a.predict(x_i)
    i += 1

# Computing Performance metrics for test data
accuracy_test_2a = accuracy_score(y_test_2a, y_pred_test_2a)
precision_test_2a = precision_score(y_test_2a, y_pred_test_2a)
recall_test_2a = recall_score(y_test_2a, y_pred_test_2a)
f1_test_2a = f1_score(y_test_2a, y_pred_test_2a)

print(f"\nEXPERIMENT - 2, PART - C (Performance Metrics for test data) :")
print(f"\nAccuracy: {accuracy_test_2a}")                           # TP+TN / (TP + FP + TN + FN)
print(f"\nPrecision: {precision_test_2a}")                         # TP / (TP+FP)
print(f"\nRecall: {recall_test_2a}")                               # TP / (TP+FN)
print(f"\nF1 Score: {f1_test_2a}\n")                               # 2*TP / (2*TP + FP + FN)


###################
###################
###################
###################
###################
###################

# EXPERIMENT - 3 :

# importing dataset - 3
input_dataset_3 = np.load('Dataset-3\inputs_Dataset-3.npy')
output_dataset_3 = np.load('Dataset-3\outputs_Dataset-3.npy')

# correcting output dataset - 3 as suggested by Professor (i.e. changing all zeroes to -1's)
output_dataset_3[output_dataset_3 == 0] = -1         # Applying Boolean Mask

X_3 = input_dataset_3    # Given input dataset 3
y_3 = output_dataset_3   # Given output dataset 3

# EXPT - 3, PART - A :

# Making an 80:20 train-test split
from sklearn.model_selection import train_test_split
X_train_3a, X_test_3a = train_test_split(X_3, test_size=0.2, shuffle=False)
y_train_3a, y_test_3a = train_test_split(y_3, test_size=0.2, shuffle=False)

perceptron_3a = Perceptron(2000)                    # Implementation of PLA for DataSet - 3 and passing the parameter 'MAX_iters' to the object 'perceptron'
perceptron_3a.fit(X_train_3a, y_train_3a)           # calling the fit method in 'perceptron' class to train PLA
perceptron_3a.plot_misclassification_history(3)     # plot the iteration Vs. number of misclassification counts

# EXPT - 3, PART - B :

# Included in the PDF Report.


# EXPT - 3, PART - C : [ calculating the values of the performance metrics for the test data ]

# Predict on train set
size_train_3a = y_train_3a.shape
y_pred_train_3a = np.zeros(size_train_3a)
i = 0
for x_i in X_train_3a:
    y_pred_train_3a[i] = perceptron_3a.predict(x_i)
    i += 1

# Calculating accuracy for train data
accuracy_train_3a = accuracy_score(y_train_3a, y_pred_train_3a)

print(f"\nCalculating accuracy for train data in EXPERIMENT - 3 :")
print(f"\nAccuracy for train data : {accuracy_train_3a}")                           # TP+TN / (TP + FP + TN + FN)


# Predict on test set
size_test_3a = y_test_3a.shape
y_pred_test_3a = np.zeros(size_test_3a )
i = 0
for x_i in X_test_3a:
    y_pred_test_3a[i] = perceptron_3a.predict(x_i)
    i += 1

# Computing Performance metrics for test data
accuracy_test_3a = accuracy_score(y_test_3a, y_pred_test_3a)
precision_test_3a = precision_score(y_test_3a, y_pred_test_3a)
recall_test_3a = recall_score(y_test_3a, y_pred_test_3a)
f1_test_3a = f1_score(y_test_3a, y_pred_test_3a)

print(f"\nEXPERIMENT - 3, PART - C (Performance Metrics for test data) :")
print(f"\nAccuracy : {accuracy_test_3a}")                               # TP+TN / (TP + FP + TN + FN)
print(f"\nPrecision : {precision_test_3a}")                             # TP / (TP + FP)
print(f"\nRecall : {recall_test_3a}")                                   # TP / (TP + FN)
print(f"\nF1 Score : {f1_test_3a}\n")                                   # 2*TP / (2*TP + FP + FN)

# EXPT - 3, PART - D :

# Included in the PDF Report.






     








    





