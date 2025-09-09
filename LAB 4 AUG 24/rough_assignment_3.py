# NAME : Shiva Ganesh Reddy Lakkasani
# ROLL NUMBER : 20EE10069
# MLFA (AI42001) Assignment - 3


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

##################
##################

# np.random.seed(42)

# Load the dataset
data = pd.read_csv('.\Iris.csv')

# display initial data insights
print(data)
print(data.shape)
print(data.describe())
print(data.info())                           # to get a view of how many columns and what data types

##################
##################

# SPLITTING THE DATA INTO 60:20:20 RATIO 

# dropping the ID column 
data = data.drop(columns=data.columns[0])

# splitting data into features (X) and target (y)
X = data.drop(columns='Species')
y = data['Species']



# 80% for [training and validation], 20% for [testing]
# stratify = y makes sure that the train_valid and test datasets have the same proportion of class labels as the original dataset
X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 75% of 80% for training which is 60%, 25% of 80% for validation which is 20%
# stratify = y_train_val makes sure that the train and valid datasets have the same proportion of class labels as the original dataset
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=0.25, random_state=42, stratify=y_train_validation)

# display data insights after splitting
print(data)
print(data.info()) 
print("Training dataset:", X_train.shape, y_train.shape)
print("Validation dataset:", X_validation.shape, y_validation.shape)
print("Testing dataset:", X_test.shape, y_test.shape)


##################
##################


# FEATURE SCALING 

scaler = StandardScaler()

# fit on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# only transform validation and test data
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

# adding a bias term (column of ones) to the beginning of X
X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_validation_scaled = np.hstack((np.ones((X_validation_scaled.shape[0], 1)), X_validation_scaled))
X_test_scaled = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))


##################
##################

# ONE-HOT ENCODING OF y_train, y_validation and y_test :

def one_hot_encode(y, class_mapping=None):

    if class_mapping is None:
        # new mapping from class label to integer index
        class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}

        # class mapping : 
        # {
        #     'Iris-setosa': 0,
        #     'Iris-versicolor': 1,
        #     'Iris-virginica': 2
        # }

    
    # converting class labels to integers using the mapping
    y_int = np.array([class_mapping[label] for label in y])
    
    # perform one-hot encoding
    one_hot_encoded_y = np.eye(len(class_mapping))[y_int]
    
    return one_hot_encoded_y, class_mapping

# one-hot encode y_train and get the class mapping
y_train_one_hot, class_mapping = one_hot_encode(y_train)

# one-hot encode y_validation and y_test using the same class mapping
y_validation_one_hot, _ = one_hot_encode(y_validation, class_mapping=class_mapping)
y_test_one_hot, _ = one_hot_encode(y_test, class_mapping=class_mapping)


# print('\ny_train_one_hot vector : ')
# print(y_train)
# print(y_train_one_hot)
# print('\ny_validation_one_hot vector : ')
# print(y_validation_one_hot)
# print('\ny_test_one_hot vector : ')
# print(y_test_one_hot)

##################
##################

# IMPLEMENTATION OF GRADIENT DESCENT APPROACH TOWARDS LOGISTIC REGRESSION FOR MULTIPLE CLASSES [LOG_MUL_GRAD] WITH MINIBATCH :

# softmax function is used in multi-class classification problems to convert logits into probabilities
def softmax(logits): 
    numerator = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # subtracting the maximum logit value for each row from every logit value for that instance fo numerical stability
    # numerator = e^{logit(particular instance)}
    return numerator / np.sum(numerator, axis=1, keepdims=True)    # return the softmax function, p(i) = P(C(i)/x) = e^(logit of x for class 'i') / [sum of e^(logit of x for each j = 1,2,3,....k)]

def cross_entropy(y_mini_batch, y_probability):
    # m = len(y_mini_batch) [NOT REQUIRED]
    return -np.mean(np.sum(y_mini_batch * np.log(y_probability) + 1e-10, axis=1))   # added a small constant in the log calculation to avoid log(0)

def compute_gradients(X, y, y_probability): # dimensions of X : number_of_inputs * number_of_features ; dimensions of y, y_probability : number_of_inputs * number_of_classes
    m = len(X)
    # gradient descent = [ sum over number_of_inputs( [yj^(n) - pj,n] * x_n ) ]
    return X.T.dot(y_probability - y)                     

def get_mini_batch(X, y, mini_batch_index, mini_batch_size):
    begin = mini_batch_index * mini_batch_size
    end = begin + mini_batch_size
    return X[begin:end], y[begin:end]

def log_mul_grad(X, y, number_of_classes, number_of_features, eta, number_of_epochs=50, mini_batch_size=30): # eta is the learning rate

    # converting y into one-hot encoding vector, where 'i'th entry of 'y' is 1 if the datapoint belongs to the 'i'th class; else 0
    # DIMENSIONS : number_of_inputs[instances] * number_of_classes
    # y_one_hot_vector, _ = one_hot_encode(y, class_mapping=class_mapping)

    # random weight initialization : 
    # theta has dimensions (number of features × number of classes)
    # each column in theta represents the weights for a particular class
    # DIMENSIONS : number_of_features * number_of_classes
    theta = np.random.randn(number_of_features, number_of_classes)
    # print(theta)

    # an epoch is one forward and backward pass of all training examples
    for epoch in range(number_of_epochs):
        shuffled_indices = np.random.permutation(len(X))             # training data is shuffled to ensure that the mini-batches change in every epoch
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for mini_batch_index in range(0, len(X), mini_batch_size):        # in every epoch, the dataset is divided into mini-batches of size mini_batch_size = 30
            
            X_mini_batch, y_mini_batch = get_mini_batch(X_shuffled, y_shuffled, mini_batch_index, mini_batch_size)         # fetches a mini-batch of the data
            
            if not len(X_mini_batch):
                continue  # skip this iteration if the mini-batch is empty

            # DIMENSIONS of logits : number_of_inputs[instances] * number_of_classes
            logits = X_mini_batch.dot(theta)                              # for the given mini-batch, calculating logits = X.theta  for each class           
            y_probability = softmax(logits)                                 # logits are passed through softmax function to get probability of an instance belonging to a particular class
            
            # error function is log-likelihood also called cross-entropy given by,
            # - [sum over n[sum over i[ yi^(n) * log(pi,n) ]]]
            loss = cross_entropy(y_mini_batch, y_probability)                    # cross-entropy loss is calculated between true one-hot encoded labels [y_batch = yi's] and predicted probabilities[y_probability = pi's]
            gradients = compute_gradients(X_mini_batch, y_mini_batch, y_probability)  # the gradient of the loss with respect to the model weights which indicates the direction and magnitude of change needed in weights to reduce the error
            theta = theta - eta * gradients                          # model weights are updated using computed gradients and the learning rate [eta]

    return theta

##################
##################

# EVALUATION OF THE MODEL BY CALCULATING PERFORMANCE METRICS  :

def calculate_performance_metrics(y_true, y_predict):
    classes = np.unique(y_true)
    performance_metrics = {}

    for each_class in classes:
        # TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative
        TP = np.sum((y_true == each_class) & (y_predict == each_class))
        TN = np.sum((y_true != each_class) & (y_predict != each_class))
        FP = np.sum((y_true != each_class) & (y_predict == each_class))
        FN = np.sum((y_true == each_class) & (y_predict != each_class))

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0       # harmonic mean of recall and precision

        performance_metrics[each_class] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score
        }

    return performance_metrics


# function for computing confusion matrix : 

def compute_confusion_matrix(y_true, y_pred):
    classes = sorted(list(set(y_true)))                                # to set the uniform order of rows and columns for every code execution
    matrix = {c: {d: 0 for d in classes} for c in classes}
    
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    return matrix

##################
##################
##################
##################
##################



# EXPERIMENT - 1 : 

# [ Performance values for model WITH FEATURE SCALING on VALIDATION DATA SET ]

number_of_features = X_train_scaled.shape[1]                                      # number of features including bias term in the dataset i.e. 5
number_of_classes = len(np.unique(y_train))                                       # unique number of classes in y_train i.e. 3  (iris-setosa, iris-versicolor, and iris-virginica)

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
accuracies_with_scaling_in_percentage = []

accuracies_with_scaling = []
precisions_with_scaling = []
recalls_with_scaling = []
f1_scores_with_scaling = []

for eta in learning_rates:

    # log_mul_grad function trains a logistic regression model using gradient descent for multi-class classification
    theta = log_mul_grad(X_train_scaled, y_train_one_hot, number_of_classes, number_of_features , eta)
    
    logits = X_validation_scaled.dot(theta)       # calculating logits = X_validation_scaled.theta  for each class 
    y_probability = softmax(logits)               # each entry in this matrix indicates the probability of an instance belonging to a particular class
    y_predict = np.argmax(y_probability, axis=1)  # predicts the class with the highest probability for each instance or data point
    
    
    # print("y_probability:", y_probability)
    # print('\ny_validation')
    # print(y_validation)
    # print(y_validation.shape)
    # print('\ny_predict')
    # print(y_predict)
    # print(y_predict.shape)
    
    y_validation_numeric = y_validation.map(class_mapping).values                 # convert y_validation into the vector whose values are 0,1,2 based on the class_mapping, to calculate the precision
    metrics_with_scaling = calculate_performance_metrics(y_validation_numeric, y_predict)      # calculate_performance_metrics function returns metrics like accuracy, precision, recall, and f1 score for each class
    
    accuracy_with_scaling = np.mean([metrics_with_scaling[each_class]["Accuracy"] for each_class in np.unique(y_validation_numeric)])        #  calculates the average accuracy over all classes
    precision_with_scaling = np.mean([metrics_with_scaling[each_class]["Precision"] for each_class in np.unique(y_validation_numeric)])
    recall_with_scaling = np.mean([metrics_with_scaling[each_class]["Recall"] for each_class in np.unique(y_validation_numeric)])
    f1_score_with_scaling = np.mean([metrics_with_scaling[each_class]["F1 Score"] for each_class in np.unique(y_validation_numeric)])


    accuracies_with_scaling_in_percentage.append(accuracy_with_scaling*100)     # appending accuracies in %
    
    accuracies_with_scaling.append(accuracy_with_scaling)
    precisions_with_scaling.append(precision_with_scaling)
    recalls_with_scaling.append(recall_with_scaling)
    f1_scores_with_scaling.append(f1_score_with_scaling)


# printing the performance value of the model WITH FEATURE SCALING for different learning rates : 

for i, lr in enumerate(learning_rates):
    print(f'\n[EXPT - 1] Performance Metrics of the model WITH FEATURE SCALING for learning rate = {lr} :')
    print(f'Accuracy: {accuracies_with_scaling[i]:.2f}')
    print(f'Precision: {precisions_with_scaling[i]:.4f}')
    print(f'Recall: {recalls_with_scaling[i]:.4f}')
    print(f'F1 Score: {f1_scores_with_scaling[i]:.4f}')



# plotting Validation Accuracy (in %) vs Learning Rate [WITH FEATURE SCALING]

plt.figure(figsize=(15,11))
plt.plot(learning_rates, accuracies_with_scaling_in_percentage, 'o-')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy (in %)')
plt.title('[EXPT - 1] Validation Accuracy (in %) vs Learning Rate [WITH FEATURE SCALING on VALIDATION DATA SET]')
plt.grid(True)
plt.show()

best_hyperparameter_learning_rate = learning_rates[np.argmax(accuracies_with_scaling_in_percentage)]
print(f"\n\n[EXPT - 1] Best value of Hyperparameter Learning Rate WITH FEATURE SCALING on VALIDATION DATA SET : {best_hyperparameter_learning_rate}\n\n")



##################
##################
##################
##################
##################


# EXPERIMENT - 2 : 


# segregating the training data by class label

data_set_setosa = X_train_scaled[y_train == 'Iris-setosa']
data_set_versicolor = X_train_scaled[y_train == 'Iris-versicolor']
data_set_virginica = X_train_scaled[y_train == 'Iris-virginica']

# log_mul_grad_probabilities function to save mean probabilities after each epoch

def log_mul_grad_probabilities(X, y, number_of_classes, number_of_features, eta, number_of_epochs=50, mini_batch_size=30):
    theta = np.random.randn(number_of_features, number_of_classes)
    mean_probabilities = []                                              # to store mean probabilities after every epoch

    for epoch in range(number_of_epochs):
        shuffled_indices = np.random.permutation(len(X))
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for mini_batch_index in range(0, len(X), mini_batch_size):        
            X_mini_batch, y_mini_batch = get_mini_batch(X_shuffled, y_shuffled, mini_batch_index, mini_batch_size)
            if not len(X_mini_batch):
                continue
            logits = X_mini_batch.dot(theta)
            y_probability = softmax(logits)
            loss = cross_entropy(y_mini_batch, y_probability)
            gradients = compute_gradients(X_mini_batch, y_mini_batch, y_probability)
            theta = theta - eta * gradients
        
        logits = X.dot(theta)
        y_probability = softmax(logits)
        mean_probabilities.append(np.mean(y_probability, axis=0))

    return mean_probabilities



# function for plotting the average class probability vs epochs

def plot_mean_probabilities(data_sub_set, class_label):
    mean_probabilities = log_mul_grad_probabilities(data_sub_set, y_train_one_hot[np.where(y_train == class_label)], number_of_classes, number_of_features, best_hyperparameter_learning_rate)
    
    # print(y_train_one_hot[np.where(y_train == class_label)])
    # print(mean_probabilities)
    plt.figure(figsize=(16,10))
    
    for i, label in enumerate(class_mapping.keys()):
        probs = [mean_prob[i] for mean_prob in mean_probabilities]
        plt.plot(probs, label=label)
    
    plt.title(f'[EXPT - 2] Mean Probabilities vs Epochs for {class_label}')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Probability')
    plt.legend()
    plt.grid(True)
    plt.show()



# call plot_mean_probabilities for each sub-data set

plot_mean_probabilities(data_set_setosa, 'Iris-setosa')
plot_mean_probabilities(data_set_versicolor, 'Iris-versicolor')
plot_mean_probabilities(data_set_virginica, 'Iris-virginica')


##################
##################
##################
##################
##################



# EXPERIMENT - 3 : 


# TRAINING THE MODEL 
theta_exp_3 = log_mul_grad(X_train_scaled, y_train_one_hot, number_of_classes, number_of_features, best_hyperparameter_learning_rate)


# [EXPT - 3, PART - A]: PREDICTING THE CLASSES FOR THE TEST DATASET :

logits_test = X_test_scaled.dot(theta_exp_3)
y_test_probability = softmax(logits_test)
y_test_predictions = np.argmax(y_test_probability, axis=1)
y_test_predicted_labels = [list(class_mapping.keys())[i] for i in y_test_predictions]
y_test_numeric = y_test.map(class_mapping).values

# print(y_test_probability)
# print(y_test_predictions)
# print(y_test_predictions)
# print(y_test_numeric)


# CONFUSION MATRIX FOR THE TEST DATASET :

confusion_matrix_test = compute_confusion_matrix(y_test, y_test_predicted_labels)



# plotting the confusion matrix as heatmap FOR THE TEST DATASET
plt.figure(figsize=(15,11))
sns.heatmap(pd.DataFrame(confusion_matrix_test), annot=True, cmap="YlGnBu")
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('[EXPT - 3] Confusion Matrix and Heatmap for TEST DATA SET')
plt.show()


# PERFORMANCE METRICS FOR EACH CLASS in THE TEST DATASET :
calculated_performance_metrics_test = calculate_performance_metrics(y_test_numeric, y_test_predictions)


for class_index, metrics in calculated_performance_metrics_test.items():
    
    class_index_labels = list(class_mapping.keys())[class_index]
    print(f"[EXPT - 3] Performance metrics for Class in TEST DATA SET: {class_index_labels}\n")
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("\n")

##################
##################

# [EXPT - 3, PART - B]: PREDICTING THE CLASSES FOR THE VALIDATION DATASET :

logits_validation = X_validation_scaled.dot(theta_exp_3)
y_validation_probability = softmax(logits_validation)
y_validation_predictions = np.argmax(y_validation_probability, axis=1)
y_validation_predicted_labels = [list(class_mapping.keys())[i] for i in y_validation_predictions]
y_validation_numeric = y_validation.map(class_mapping).values

# print(y_validation_probability)
# print(y_validation_predictions)
# print(y_validation_predictions)
# print(y_validation_numeric)


# CONFUSION MATRIX :

confusion_matrix_validation = compute_confusion_matrix(y_validation, y_validation_predicted_labels)



# plotting the confusion matrix as heatmap for VALIDATION DATA SET
plt.figure(figsize=(15,11))
sns.heatmap(pd.DataFrame(confusion_matrix_validation), annot=True, cmap="YlGnBu")
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('[EXPT - 3] Confusion Matrix and Heatmap for VALIDATION DATA SET')
plt.show()


# PERFORMANCE METRICS FOR EACH CLASS in VALIDATION DATA SET :
calculated_performance_metrics_validation = calculate_performance_metrics(y_validation_numeric, y_validation_predictions)


for class_index, metrics in calculated_performance_metrics_validation.items():
    
    class_index_labels = list(class_mapping.keys())[class_index]
    print(f"[EXPT - 3] Performance metrics for Class in VALIDATION DATA SET: {class_index_labels}\n")
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("\n")


###########    THE END   #############