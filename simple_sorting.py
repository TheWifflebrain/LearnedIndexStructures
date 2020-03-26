# import os
import csv
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers
from keras import models
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import Sequential
import keras.backend.tensorflow_backend as KTF

from sklearn.svm import SVR
from sklearn import ensemble
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

# To use the GPU
# I get errors if I do not use this code
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

total_items = 25
# Creating the dataset
# the key or value of the data
x = []
# the position of the data
y = []
for i in range(1, (total_items) + 1):
    y.append(i)
    # as long as the function is monotonic will predict accurately
    # so num = math.sin(i) will not predict accurately as it should
    # num = math.ceil(2 ** 3 + 4 ** (2) * + 3*i * math.log(i))
    num = 8*i  # + random.randrange(0,6)
    # num = (i ** 2) - 21*(i) + 34
    x.append(num)

# y will be from 1 to 1000
# x will be num or the value of the data
[math.ceil(float(num)) for num in x]
[math.ceil(float(num)) for num in y]

# A line graph of the data
plt.figure(1)
plt.title('The data')
plt.plot(x, y)
# PDF - Probability density function
# The PDF of the data with y axis as amount in the range
plt.figure(2)
plt.hist(x, bins=20, rwidth=0.9,
         color='#607c8e')
plt.title('PDF')
plt.xlabel('x - value of the data')
plt.ylabel('y - position of the data')
# The PDF of the data with the y axis as percentages
plt.figure(3)
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.title('PDF as percentages')
plt.xlabel('x - value of the data')
plt.ylabel('y - percentage of the data occuring')
plt.hist(x, weights=np.ones(len(x)) / len(x),
         bins=20, rwidth=0.9, color='#607c8e')
# The culmative distribution function or CDF
plt.figure(4)
plt.title('CDF (∫PDF)')
plt.xlabel('x - value of the data')
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=999))
plt.ylabel('y - percentage')
plt.plot(x, y, marker='o')

# making sure there are no duplicates
duplicates = []
for item in x:
    if x.count(item) > 1:
        duplicates.append(item)
print(duplicates)

randomize = list(zip(x, y))
random.shuffle(randomize)
x, y = zip(*randomize)

# writing the x to col 1 and y to col 2 in a csv file
with open('C:/Users/black/CS499/simpleRMI/data_sort_linear.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for ix, iy in zip(x, y):
        writer.writerow([ix, iy])

# x is the key or value of the data
# y is the position
# reading data in from csv to make it more versatile
# Do not need to do this section if you just want to use the data that was just created
values = ["key", "pos"]
data = pd.read_csv(r"C:/Users/black/CS499/simpleRMI/data_sort_linear.csv", header=None, skipinitialspace=True,
                   names=values, na_values=["?"], sep=',')

data["key"] = data.key.astype(float)
data["pos"] = data.pos.astype(float)

x_data = data["key"].values
y_data = data["pos"].values

xs_data = x_data.reshape(-1, 1)
ys_data = y_data.reshape(-1, 1)

neigh = KNeighborsRegressor(n_neighbors=4)
neigh.fit(xs_data, np.ravel(ys_data))

pred_knn = neigh.predict(xs_data)
print("Prediction for KNN:", pred_knn)

# making sure there are no duplicates
list_pred = pred_knn.tolist()
duplicates = []
for item in list_pred:
    if list_pred.count(item) > 1:
        duplicates.append(item)
print("Duplicates:", duplicates)

int_pred = [math.floor(int(x)) for x in list_pred]
print("Predictions:", int_pred)

# put xs_data in position predictions
xs_data_list = xs_data.tolist()
# data_list = [int(x) for x in xs_data_list]
print("Values:", xs_data_list)

count = 0
notsortedx = []
notsortedy = []
sorted_list = [-1]*total_items
for i in range(0, total_items):
    if(sorted_list[(int_pred[i])-1] == -1):
        sorted_list[(int_pred[i])-1] = xs_data_list[i]
    else:
        notsortedx.append(xs_data_list[i])
        notsortedy.append((int_pred[i])-1)
        count = count + 1

print(sorted_list)
print(notsortedx)
print(notsortedy)

# if not sorted
# figure out what did not get inputted in the sorted list
# then sort the list with those variables in it
itr = 0
for i in range(0, count):
    placing = 1
    index = notsortedy[i]
    count = 0
    increment = 1
    while placing == 1:
        checked = 0
        print("list:", notsortedx[i])
        print("index:", index)
        print("count:", count)
        if(sorted_list[index] == -1):
            sorted_list[index] = notsortedx[i]
            placing = 0
        elif(count % 2 == 0):
            index = notsortedy[i]
            index = index + increment
            if(index >= total_items):
                index = (total_items - 1)
            checked = 1
        elif(count % 2 == 1):
            index = notsortedy[i]
            index = index - increment
            if(index < 0):
                index = 0
            checked = 1
        if(count % 2 == 0):
            increment += 1
        count += 1
        itr += 1

print(itr)
print(sorted_list)
