import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor                 # Using the SGDRegressor for LIN_MODEL_GRAD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge                        # here, Ridge is not used since, mini batch training is not possible with it, instead SGDRegressor is used



# Load the dataset
data = pd.read_csv('dataset.csv')

print(data)
print(data.shape)
print(data.describe())
print(data.info())                           # to get a view of how many columns and what data types


# Handling NaN (Missing) Values : 

# drop columns with >50% NaN values
limit = 0.5 * len(data)
data = data.dropna(thresh=limit, axis=1)     # we set axis = 1, so that dropna() operates on columns rather than rows


# for numerical features
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# impute with mean
for col in numerical_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# impute with median
#for col in numerical_cols:
#    data[col].fillna(data[col].median(), inplace=True)

# impute with forward fill
# for col in numerical_cols:
#    data[col].fillna(method='ffill', inplace=True)

# impute with backward fill
#for col in numerical_cols:
#    data[col].fillna(method='bfill', inplace=True)


# for categorical features
categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()

# impute with mode
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# impute with forward fill
#for col in categorical_cols:
#    data[col].fillna(method='ffill', inplace=True)

# impute with backward fill
#for col in categorical_cols:
#    data[col].fillna(method='bfill', inplace=True)



# Handling the Features in the dataset : 


# dropping the features as suggested in the demo
features_to_drop = ['User_ID', 'Product_ID']
data = data.drop(columns=features_to_drop)


##################
##################
##################
##################
##################


# EXPERIMENT - 1 : 

# Implementation of (EDA) Exploratory Data Analysis and Feature scaling


# distribution of features [EDA]
# data.hist(bins=65, figsize=(30,45))
# plt.suptitle("Distribution of the Features [EXPERIMENT -1]")
# plt.show()

n = len(data.columns)
n_cols = 3                                            # number of histograms per row
n_rows = int(n / n_cols) + (n % n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20,16))

for i, col in enumerate(data.columns):
    ax = axes[i//n_cols, i%n_cols]
    data[col].hist(bins=65, ax=ax)
    ax.set_title(col, fontsize=8)

plt.suptitle("Distribution of the Features [EXPERIMENT -1]", fontsize=10)
plt.tight_layout()             # spacing between subplots for better layout
plt.subplots_adjust(top=0.95)  # to ensure the suptitle doesn't overlap
plt.show()

# using one-hot encoding, to handle the non-numerical (categorical) values
features_to_encode = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years']
data = pd.get_dummies(data, columns=features_to_encode)

# converting boolean columns to binary (0 or 1)
data = data * 1

# ensuring that all columns are either float or int to implement the closed form solution
for col in data.columns:
    if data[col].dtype == 'object':  # This will check for any string columns
        raise ValueError(f"The column {col} still has non-numeric data.")
    
print("\nData after one-hot encoding : ")
print(data)


# pairwise correlation [EDA]
numerical_data = data.select_dtypes(include=[np.number])
corr_matrix = numerical_data.corr()

plt.figure(figsize=(24, 24))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap [EXPERIMENT - 1]')
plt.show()


##################
##################
##################
##################
##################


# EXPERIMENT - 2 : 



# separate features and target, here, 'Purchase' is the target. Since, the company wants to predict the target
X = data.drop(columns='Purchase')
y = data['Purchase']

# split the data into training (60%) and a temporary set (40%) 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# split the temporary set into validation and test sets, each being 20% of the main dataset to get the desired 60:20:20 split
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# making sure the split is successful for X
print("\nFor X :")
print("Actual set size:", len(X))
print("Training set size:", len(X_train))
print("Validation set size:", len(X_valid))
print("Test set size:", len(X_test))

# making sure the split is successful for y
print("\nFor y :")
print("Actual set size:", len(y))
print("Training set size:", len(y_train))
print("Validation set size:", len(y_valid))
print("Test set size:", len(y_test))


# implement the closed-form solution for linear regression
# this function calculates the parameters (theta) for linear regression using the closed-form solution 
print('\n[EXPERIMENT -2] Computing closed form solution !')

# Implementation of closed form solution approach towards linear regression [LIN_MODEL_CLOSED]

def lin_model_closed(X, y):
    X_new = np.c_[np.ones((X.shape[0], 1)), X]                            # adding a column of ones (a bias term) to the front of the matrix X
    theta = np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)         # θ=(X^(T)⋅X)^(-1)⋅X^(T)⋅y from normal equation
    return theta

# training and predicting WITHOUT scaling
theta_without_scaling = lin_model_closed(X_train, y_train)                 # calculate parameters without scaling
X_test_new = np.c_[np.ones((X_test.shape[0], 1)), X_test]                              # add bias term to the test set
predictions_without_scaling = X_test_new.dot(theta_without_scaling)                     # make predictions on the test set

# feature scaling using the StandardScaler() method
scaler = StandardScaler()                                                # initialize the standard scaler

# fit the scaler on the training data
scaler.fit(X_train)

# transform both the training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training and Predicting with scaling
theta_with_scaling = lin_model_closed(X_train_scaled, y_train)              # calculate parameters with scaled data
X_test_scaled_new = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]
predictions_with_scaling = X_test_scaled_new.dot(theta_with_scaling)

# calculating MSE [Mean Squared Error]
mse_without_scaling= mean_squared_error(y_test, predictions_without_scaling)          # compute MSE for predictions without scaling
mse_with_scaling = mean_squared_error(y_test, predictions_with_scaling)                # compute MSE for predictions with scaling

print(f"\n [EXPERIMENT -2] MSE without Scaling: {mse_without_scaling}")
print(f"\n [EXPERIMENT -2] MSE with Scaling: {mse_with_scaling}")
print(f"\n Please take a note of the above values, before the iteration count starts !")
time.sleep(10)  # Sleeps or waits for 5 seconds, to get the note of the above values


##################
##################
##################
##################
##################


# EXPERIMENT - 3 : 

# initializing the parameters 
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1]
mini_batch_size = 256
number_epochs = 50
mse_values = []



# scaling the validation data
X_valid_scaled = scaler.transform(X_valid)

it = 0           # to keep the track of number of iterations

for rate in learning_rates:
    print('\n it = ', it, 'PLEASE WAIT till it reaches 387305')
    it = it + 1

    # initializing the SGDRegressor with the current learning rate
    # the learning rate remains fixed at 'eta0' throughout training
    # 'tol' (parameter is the tolerance for the stopping criterion) is set to None, so that stopping criterion will be based solely on max_iter
    lin_model_grad = SGDRegressor(learning_rate='constant', eta0=rate, max_iter=number_epochs, tol=None)
    
    # mini-batch training
    for epoch in range(number_epochs):
        print('\n it = ', it, 'PLEASE WAIT till it reaches 387305')
        it = it + 1

        for i in range(0, len(y_train), mini_batch_size):
            print('\n it = ', it, 'PLEASE WAIT till it reaches 387305')
            it = it + 1

            X_mini = X_train_scaled[i:i+mini_batch_size]
            y_mini = y_train[i:i+mini_batch_size]
            
            lin_model_grad.partial_fit(X_mini, y_mini)
    
    # predicting on the validation set and calculating the MSE
    y_val_pred = lin_model_grad.predict(X_valid_scaled)
    mse = mean_squared_error(y_valid, y_val_pred)
    mse_values.append(mse)

# plotting MSE [Mean Squared Error] vs Learning rate
plt.figure(figsize=(12, 8))
plt.plot(learning_rates, mse_values, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.title("MSE vs Learning Rate [EXPERIMENT - 3] ")
plt.xlabel("Learning Rate")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()

# determining the best learning rate value
best_rate = learning_rates[np.argmin(mse_values)]
print(f"\n [ For EXPERIMENT - 3 ] The best learning rate is: {best_rate}")


##################
##################
##################
##################
##################


# EXPERIMENT - 4 : 


best_learning_rate = best_rate                          # use the best learning rate obtained from Experiment 3
alpha_values = np.arange(0.0, 1.1, 0.1)                 # varying ridge regression hyperparameter (alpha) from 0.0 to 1.0 with an increment of 0.1
mse_values_alpha = []


for alpha in alpha_values:
    print(f"\nTraining with alpha = {alpha}, PLEASE WAIT till aplha reaches 1.0 !")
    
    # for each alpha, ridge regression model using mini-batch approach with SGDRegressor where the penalty is 'l2' (which indicates Ridge regression)
    # here, SGDRegressor methos is used instead of 'Ridge', because we can't do mini-batch training with Ridge
    # initializing the Ridge regression with SGD (with the current alpha and best learning rate)
    # initializing the SGDRegressor with the best_learning_rate
    # the learning rate remains fixed at 'eta0' throughout training
    # 'tol' (parameter is the tolerance for the stopping criterion) is set to None, so that stopping criterion will be based solely on max_iter
    lin_model_ridge= SGDRegressor(penalty='l2', alpha=alpha, learning_rate='constant', eta0=best_learning_rate, max_iter=number_epochs, tol=None)
    
    # mini-batch training
    for epoch in range(number_epochs):
        for i in range(0, len(y_train), mini_batch_size):
            X_mini = X_train_scaled[i:i+mini_batch_size]
            y_mini = y_train[i:i+mini_batch_size]
            lin_model_ridge.partial_fit(X_mini, y_mini)
    
    # predicting on the validation set and calculating the MSE
    y_val_pred_ridge = lin_model_ridge.predict(X_valid_scaled)
    mse_ridge = mean_squared_error(y_valid, y_val_pred_ridge)
    mse_values_alpha.append(mse_ridge)

# plotting MSE vs alpha (Ridge regression hyperparameter)
plt.figure(figsize=(12, 8))
plt.plot(alpha_values, mse_values_alpha, marker='o', linestyle='-', color='b')
plt.title("MSE vs Alpha (Ridge Regression) [ For EXPERIMENT - 4 ]")
plt.xlabel("Alpha (Regularization Strength)")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()

# determining the best alpha value
best_alpha = alpha_values[np.argmin(mse_values_alpha)]
print(f"\n [ For EXPERIMENT - 4 ]  The best value of the hyperparameter alpha is: {best_alpha}")




##################
##################
##################
##################
##################

# EXPERIMENT - 5 : 



# for LIN_MODEL_CLOSED:
theta_closed_form = lin_model_closed(X_train_scaled, y_train)                         # calculate parameters using the closed-form solution
X_test_scaled_new = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]        # add bias term to the test set

predictions_closed_form = X_test_scaled_new.dot(theta_closed_form)                    # make predictions on the test set
mse_closed_form = mean_squared_error(y_test, predictions_closed_form)                 # compute MSE for the closed-form solution

# for LIN_MODEL_GRAD (using best learning rate) :
lin_model_grad_optimal = SGDRegressor(learning_rate='constant', eta0=best_rate, max_iter=number_epochs, tol=None)
lin_model_grad_optimal.fit(X_train_scaled, y_train)
predictions_grad = lin_model_grad_optimal.predict(X_test_scaled)
mse_grad = mean_squared_error(y_test, predictions_grad)

# for LIN_MODEL_RIDGE (using best learning rate and best alpha):
lin_model_ridge_optimal = SGDRegressor(penalty='l2', alpha=best_alpha, learning_rate='constant', eta0=best_rate, max_iter=number_epochs, tol=None)
lin_model_ridge_optimal.fit(X_train_scaled, y_train)
predictions_ridge = lin_model_ridge_optimal.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, predictions_ridge)

print(f"\n [ For EXPERIMENT - 5 ] : ")

print(f"MSE for LIN_MODEL_CLOSED: {mse_closed_form}")
print(f"MSE for LIN_MODEL_GRAD: {mse_grad}")
print(f"MSE for LIN_MODEL_RIDGE: {mse_ridge}")







