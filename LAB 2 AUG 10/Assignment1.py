# Name : Shiva Ganesh Reddy Lakkasani
# Roll Number : 20EE10069
# MLFA (AI42001) Assignment - 1

# importing dataset - 1
import numpy as np
input_dataset_1 = np.load('Dataset-1\inputs_Dataset-1.npy')

output_dataset_1 = np.load('Dataset-1\outputs_Dataset-1.npy')



# correcting output dataset - 1 (i.e. changing all zeroes to 1's)
output_dataset_1[output_dataset_1 == 0] = -1         # Applying Boolean Mask


# Implementation of PLA for DataSet - 1 :

X = input_dataset_1    # Given input dataset 1
y = output_dataset_1   # Given output dataset 1

# defining a class called "Perceptron"
class Perceptron:
    def __init__(self, MAX_iters) :
        self.MAX_iters = MAX_iters
        
    def predict(self, inputs):
        return 1 if np.dot(inputs, self.weights) >= 0 else -1
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        #initialize the parameters
        self.weights = np.random.uniform(-5.0, 5.0, n_features)  # Here, n_features includes the extra dimension for '1', hence, we are not doing, 'n_features+1'

        
        for _ in range(self.MAX_iters):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                if prediction != label:
                    self.weights += label * inputs

        # if(label == y):
        #     print("This is linearly seperable data \n")
        # else : 
        #     print("This is not linearly seperable data")


perceptron = Perceptron(2000)       # passing the parameter 'MAX_iters' to the object 'perceptron'
perceptron.fit(X, y)                # calling the fit method in 'perceptron' class 

# EXPERIMENT - 1 :

# PART - A :

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

for i in range(10):
    k = i + 2
    kf = KFold(n_splits=k)

    knn_kfold = []
    prediction_kfold = []
    metrics_kfold = []

    for train_index, test_index in kf.split(X):      # this loop will run for 'k' times
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn_kfold.append(KNeighborsClassifier(n_neighbors=5))
        knn_kfold[-1].fit(X_train, y_train)
        prediction = knn_kfold[-1].predict(X_test)

        acc = accuracy_score(y_test, prediction)     # TP+TN / (TP + FP + TN + FN)
        prec = precision_score(y_test, prediction)   # TP / (TP+FP)
        rec = recall_score(y_test, prediction)       # TP / (TP+FN)
        f1 = f1_score(y_test, prediction)            # (2 * prec * rec)/(prec + rec)

        prediction_kfold.append(prediction)
        metrics_kfold.append([acc, prec, rec, f1])
        

    for metrics in metrics_kfold:
        print(metrics)

    #find mean and variance for performance metrics of the models
    metrics_array = np.array(metrics_kfold)
    mean_metrics = np.mean(metrics_array, axis=0)
    var_metrics = np.var(metrics_array, axis=0)

    print("\nMean of the metrics: ", mean_metrics)
    print("\nVariance of the metrics: ", var_metrics)


# PART - C : [80:20 split and counting the number of misclassified instances]
print(X)
from sklearn.model_selection import train_test_split
train, test = train_test_split(X, test_size=0.2, shuffle=False)



    





