# **idea is from
# https://arxiv.org/abs/1712.01208
# **inspiration for this code is from
# https://github.com/Learned-Index-Structure/LIS-Training/blob/master/Example.ipynb

# making the margins wider in jupyter
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import os
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from keras import layers
from keras import models
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import numpy as np
from matplotlib.ticker import PercentFormatter
import math
import random
import matplotlib.ticker as mtick
##from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

# Creating a linear dataset with a little bit of randomness
# the key or value of the data
x = []
# the position of the data
y = []
for i in range(1, 1000):
    y.append(i)
    # + random.randrange(0,1)
    #num = math.ceil(2 ** 3 + 4 ** (2) * + 3*i * math.log(i))
    #num = 8*i + random.randrange(0,6)
    num = (i ** 2) - 21*(i) + 34
    #num = math.sin(i)
    x.append(num)

[float(num) for num in x]
[math.ceil(float(num)) for num in y]
# x will be from 1 to 500
# y will be num

############
plt.figure(1)
plt.plot(x, y)
############
plt.figure(2)
plt.hist(x, bins=20, rwidth=0.9,
         color='#607c8e')
plt.title('PDF')
plt.xlabel('x - value of the data')
plt.ylabel('y - position of the data')
###########
plt.figure(3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('PDF as percentages')
plt.xlabel('x - value of the data')
plt.ylabel('y - percentage of the data occuring')
plt.hist(x, weights=np.ones(len(x)) / len(x),
         bins=20, rwidth=0.9, color='#607c8e')
##########
# The culmative distribution function
plt.figure(4)
plt.title('CDF (∫PDF)')
plt.xlabel('x - value of the data')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=999))
plt.ylabel('y - percentage')
plt.plot(x, y, marker='o')
##########
# making sure there are no duplicates
# also need to check to see if the data is sorted (I made the equation so that it would be sorted)
duplicates = []
for item in x:
    if x.count(item) > 1:
        duplicates.append(item)
print(duplicates)
############
# writing the x to col 1 and y to col 2 in a csv file

with open('data2.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    for ix, iy in zip(x, y):
        writer.writerow([ix, iy])
##############
# x is the key
# y is the pos or value
# reading data in from csv to make it more versatile
# Do not need to do this section if you just want to use the data that was just created

values = ["key", "pos"]
data = pd.read_csv(r"C:/Users/black/CS499/simpleRMI/data2.csv", header=None, skipinitialspace=True,
                   names=values, na_values=["?"], sep=',')

data["key"] = data.key.astype(float)
data["pos"] = data.pos.astype(float)

x_data = data["key"].values
y_data = data["pos"].values

# if you want to do a train test split
#X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.0001)
#################
# building the neural network

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


pre_pos_ML = []

key = 36214
# position of the key you are trying to predict
key_pos_list = 200
# if you want to test multiple x(key) values at a time
#ex. array = np.array([0,100,101])
array = np.array([key_pos_list])
# if you want to test multiple x(key) values at a time
test_key = key  # x_data[array]
array = [key]
print("The key(s) you are testing:", test_key)


# start of NN
all_nn = []
for i in range(0, 1):
    def build_model():
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu', input_shape=(1,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1))
        # found better results with mae than mse
        model.compile(optimizer='adam',
                      loss='mae',
                      metrics=['mae'])
        return model

    model = build_model()
    model.fit(x_data, y_data, epochs=200, batch_size=10, verbose=0)
    mse_score, mae_score = model.evaluate(x_data, y_data)
    all_nn.append([mae_score, mse_score, model.predict_on_batch(array)])


def getKey(item):
    return item[0]


sorted(all_nn, key=getKey)
# mae - mean absolute error
# mse - mean squared error
print(all_nn[0][2])
###################

for i in range(0, 1):
    print("Pred:", all_nn[i])

# getting the predictions from the NN
f_pred_nn = model.predict_on_batch(array)
print("The predictions for the position of the key(s) are:", f_pred_nn)

# getting one x(key) value to run the linear models on
# this value will now go to stage 2 of the RMI
fine_tune_pred = math.ceil(f_pred_nn)
print("The prediction for the position of the key that is going to the linear model:", fine_tune_pred)

pre_pos_ML.append(f_pred_nn)

# for grid search
# https://medium.com/datadriveninvestor/an-introduction-to-grid-search-ff57adcc0998
# SVM
xs_data = x_data.reshape(-1, 1)
ys_data = y_data.reshape(-1, 1)
clf_gs = GridSearchCV(
    estimator=SVR(kernel='rbf'),
    param_grid={
        'C': [0.1, 1, 100, 1000],
        'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
        'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
    },
    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
clf_gs.fit(xs_data, np.ravel(ys_data))
print(clf_gs.predict([[key]]))
pred_svm = clf_gs.predict([[key]])
pre_pos_ML.append(pred_svm)
# print(clf_gs)

print(clf_gs.best_score_)
# print(sorted(clf_gs.cv_results_.keys()))
print("Best C value:", clf_gs.best_estimator_.C)
print("Best epsilon value:", clf_gs.best_estimator_.epsilon)
print("Best gamma value:", clf_gs.best_estimator_.gamma)

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
# KNN
neigh = KNeighborsRegressor(n_neighbors=3)
neigh.fit(xs_data, np.ravel(ys_data))

print(neigh.predict([[key]]))
pred_knn = neigh.predict([[key]])

pre_pos_ML.append(pred_knn)

# https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html
# Decision Tree
# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=1)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=10)
regr_1.fit(xs_data, np.ravel(ys_data))
regr_2.fit(xs_data, np.ravel(ys_data))
regr_3.fit(xs_data, np.ravel(ys_data))

# Predict
X_test = np.arange(key)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

pre_pos_ML.append(y_1[-1])
pre_pos_ML.append(y_2[-1])
pre_pos_ML.append(y_3[-1])

print("Using max depth 1:", y_1[-1], "\nThe entire decisions:", y_1)
print("Using max depth 5:", y_2[-1], "\nThe entire decisions:", y_2)
print("Using max depth 10:", y_3[-1], "\nThe entire decisions:", y_3)

# Plot the results
plt.figure(5)
plt.scatter(xs_data, np.ravel(ys_data), s=20,
            edgecolor="black", c="orange", label="data")
plt.plot(X_test, y_1, color="blue", label="max_depth=1", linewidth=2)
plt.plot(X_test, y_2, color="green", label="max_depth=5", linewidth=2)
plt.plot(X_test, y_3, color="red", label="max_depth=10", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
# Bayesian Ridge
clf_r = linear_model.BayesianRidge()
clf_r.fit(xs_data, np.ravel(ys_data))

print(clf_r.predict([[key]]))
pred_br = clf_r.predict([[key]])
pre_pos_ML.append(pred_br)


# https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
# Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf_gb = ensemble.GradientBoostingRegressor(**params)

clf_gb.fit(xs_data, np.ravel(ys_data))
mse_gb = mean_squared_error(np.ravel(ys_data), clf_gb.predict(xs_data))
print("MSE: %.4f" % mse_gb)
print("Predicted position for", key, " ", clf_gb.predict([[key]]))


pred_gb = clf_gb.predict([[key]])
pre_pos_ML.append(pred_gb)

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(clf_gb.staged_predict(xs_data)):
    test_score[i] = clf_gb.loss_(np.ravel(ys_data), y_pred)

plt.figure(6)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf_gb.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


# NN
# SVM
# KNN
# Decision Tree 1,2,3
# Bayesian Ridge
# Gradient Boosting
###########################################################################################################


print(pre_pos_ML)


# how many different buckets
buckets = 100

# splitting the data up evenly into the number of buckets specified
# ex. if data from 1 to 100 and 10 specificed buckets
# data 1 to 10 in bucket 0
# 11 to 20 in bucket 1 ...
# 91-100 in bucket 9
xdata = np.array_split(x_data, buckets)
ydata = np.array_split(y_data, buckets)


print("***********This is the pre_pos_ML", pre_pos_ML,
      "*********************************************************")
ml_algo = ["NN", "SVM", "KNN", "DT1", "DT2", "DT3", "BR", "GB"]
num_it = []
counter = 0
fig_num = 7
for fine_tune_pred in pre_pos_ML:
    print("\n\n\nNow using", ml_algo[counter])
    for i in range(0, buckets):
        # seeing which bucket will the NN prediction fall into
        if((fine_tune_pred >= ydata[i][0]) & (fine_tune_pred <= ydata[i][-1])):
            print("The values in the", i, "bucket range from:",
                  ydata[i][0], "to:", ydata[i][-1])
            print("The", ml_algo[counter], " predicted:", fine_tune_pred)
            print("The prediction falls under bucket:", i)
            in_bucket = i

    # creating a linear regression model on the dataset in the chosen bucket

    # linear regression model
    re_x_data = xdata[in_bucket].reshape(-1, 1)
    re_y_data = ydata[in_bucket].reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(re_x_data, re_y_data)

    # printing out the equation for this bucket with the key you wanted to test
    print("The equation for this bucket", regressor.coef_,
          "* X", "+", regressor.intercept_)
    print("The equation for the specified key",
          regressor.coef_, "*", key, "+", regressor.intercept_)
    predicted_position = regressor.coef_ * key + regressor.intercept_
    print("The linear model predicted the position:", predicted_position)
    actual = math.ceil(y_data[key_pos_list])
    print("The actual position of the key:", actual)

    # the actual y max/min value for the bucket
    y_max = re_y_data[-1]
    y_min = re_y_data[0]
    # getting the predictions for each x(key) value for the bucket
    error = regressor.predict(re_x_data)
    # getting how far off the linear regression prediction was off from the actual values
    diff = np.subtract(re_y_data, error)

    print("The actual y max value for the bucket:", y_max)
    print("The actual y min value for the bucket:", y_min)
    print("The min error that the linear regression predicted:", np.min(diff))
    print("The max error that the linear regression predicted:", np.max(diff))

    # plotting the bucket data
    # the actual data points are the circles
    # the function that the linear regression predicted in the dotted line
    # it is 0 or 1 because in my testing NN can predict the wrong bucket and
    # the rest ml algo predict the pretty much the same bucket
    if(counter == 0 or counter == 1):
        plt.figure(fig_num)
        #plt.title("ML algorithm:", ml_algo[counter])
        plt.clf()
        plt.plot(re_x_data, re_y_data, 'o', label='Actual Data', alpha=1.0)
        plt.plot(re_x_data, error, '--', label='Regression', alpha=1.0)
        plt.legend(loc='best')
    fig_num = fig_num + 1
    counter = counter + 1

    # using exponential search to search for the actual y(position) value
    # could possibly try binary search with the ends being the min and max errors for larger datasets
    counting = 0
    upcount = 1
    downcount = 1
    searching = 1
    n_pp = math.ceil(predicted_position)
    #n_pp = math.ceil(predicted_position + np.max(diff))
    #print("Adding in the error bound:", n_pp)
    while searching == 1:
        if(n_pp == actual):
            searching = 0
        elif(n_pp < actual):
            n_pp += upcount
            upcount *= 2
            downcount = 1
        elif(n_pp > actual):
            n_pp -= downcount
            downcount *= 2
            upcount = 1
        counting += 1
    print("It took", counting, "iterations to find the actual position value")
    num_it.append(counting)

print("\n\n")
# printing out the equation for each bucket using linear regression
for i in range(0, buckets):
    re_x_data = xdata[i].reshape(-1, 1)
    re_y_data = ydata[i].reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(re_x_data, re_y_data)
    print("Bucket", i, ":", regressor.coef_, "* X +", regressor.intercept_)

print("\n\n")
for i in range(0, 8):
    print(ml_algo[i], "took", num_it[i], "iterations")

plt.show()
