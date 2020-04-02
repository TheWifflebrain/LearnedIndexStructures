import os
import csv
import math
import time
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# https://www.growingwiththeweb.com/2015/06/bucket-sort.html
# bucket sort


def bucketSort(array):
    bucketSize = 10
    if len(array) == 0:
        return array

    # Determine minimum and maximum values
    minValue = array[0]
    maxValue = array[0]
    for i in range(1, len(array)):
        if array[i] < minValue:
            minValue = array[i]
        elif array[i] > maxValue:
            maxValue = array[i]

    # Initialize buckets
    bucketCount = math.floor((maxValue - minValue) / bucketSize) + 1
    buckets = []
    for i in range(0, bucketCount):
        buckets.append([])

    # Distribute input array values into buckets
    for i in range(0, len(array)):
        buckets[math.floor((array[i] - minValue) /
                           bucketSize)].append(array[i])

    # Sort buckets and place back into input array
    array = []
    for i in range(0, len(buckets)):
        insertionSort(buckets[i])
        for j in range(0, len(buckets[i])):
            array.append(buckets[i][j])

    return array

# https://www.programiz.com/dsa/radix-sort
# radix sort


def countingSort(array, place):
    size = len(array)
    output = [0] * size
    count = [0] * 10

    for i in range(0, size):
        index = array[i] // place
        count[index % 10] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = size - 1
    while i >= 0:
        index = array[i] // place
        output[count[index % 10] - 1] = array[i]
        count[index % 10] -= 1
        i -= 1

    for i in range(0, size):
        array[i] = output[i]


def radixSort(array):
    max_element = max(array)
    place = 1
    while max_element // place > 0:
        countingSort(array, place)
        place *= 10
# heap sort
# https://www.geeksforgeeks.org/heap-sort/


def heapify(arr, n, i):
    largest = i  # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root
    if l < n and arr[i] < arr[l]:
        largest = l

    # See if right child of root exists and is
    # greater than root
    if r < n and arr[largest] < arr[r]:
        largest = r

    # Change root, if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap

        # Heapify the root.
        heapify(arr, n, largest)

# The main function to sort an array of given size


def heapSort(arr):
    n = len(arr)

    # Build a maxheap.
    for i in range(n, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  # swap
        heapify(arr, i, 0)

# https://www.geeksforgeeks.org/python-program-for-bubble-sort/
# bubble sort


def bubbleSort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):

        # Last i elements are already in place
        for j in range(0, n-i-1):

            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# https://www.geeksforgeeks.org/insertion-sort/
# insertion sort


def insertionSort(arr):

    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# https://www.geeksforgeeks.org/python-program-for-quicksort/
# quick sort


def partition(arr, low, high):
    i = (low-1)         # index of smaller element
    pivot = arr[high]     # pivot

    for j in range(low, high):

        # If current element is smaller than or
        # equal to pivot
        if arr[j] <= pivot:

            # increment index of smaller element
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i+1], arr[high] = arr[high], arr[i+1]
    return (i+1)

# Function to do Quick sort


def quickSort(arr, low, high):
    if low < high:

        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)

        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi-1)
        quickSort(arr, pi+1, high)


total_items = 1000
# Creating the dataset
# the key or value of the data
x = []
# the position of the data
y = []
for i in range(1, (total_items) + 1):
    y.append(i)
    # as long as the function is monotonic will predict accurately
    # so num = math.sin(i) will not predict accurately as it should
    num = math.ceil(2 ** 3 + 4 ** (2) * + 3*i * math.log(i))
    # num = 10*i  # + random.randrange(0,6)
    #num = (i ** 2) - 21*(i) + 34
    x.append(num)

# y will be from 1 to total_items
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
# print(duplicates)

# randomizing the data
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
# print(x_data)

# making predictions for ml
tML = time.perf_counter()
# KNN
#neigh = KNeighborsRegressor(n_neighbors=500)
#neigh.fit(xs_data, np.ravel(ys_data))
# DT
neigh = DecisionTreeRegressor(max_depth=9)
neigh.fit(xs_data, np.ravel(ys_data))

# start time for ml
t0 = time.perf_counter()
pred_knn = neigh.predict(xs_data)
#print("Prediction for KNN:", pred_knn)

list_pred = pred_knn.tolist()
# print(list_pred)

# making sure there are no duplicates
# duplicates = []
# for item in list_pred:
#     if list_pred.count(item) > 1:
#         duplicates.append(item)
#print("Duplicates:", duplicates)

# making everything to an int
int_pred = [math.floor(int(x)) for x in list_pred]
#print("Predictions:", int_pred)

# put xs_data in to a list
xs_data_list = xs_data.tolist()
#print("Values:", xs_data_list)

# putting in the predicted values into the correct position
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
# print(sorted_list)
# print(notsortedx)
# print(notsortedy)

# if already sorted then the notsorted list will have length 0 and is finished sorting
first_sorted = 0
if len(notsortedx) == 0:
    t1 = time.perf_counter() - t0
    first_sorted = 1
else:
    # if not sorted
    # figure out what did not get inputted in the sorted list
    # then sort the list with those variables in it
    # index will change +1 from predicted index then to -1 from predicted index
    # then +2, -2, +3, -3 etc. until it has placed all non sorted values
    # is there a better way?
    # itr = 0
    for i in range(0, count):
        placing = 1
        index = notsortedy[i]
        count = 0
        increment = 1
        while placing == 1:
            checked = 0
            # print(notsortedy[i])
            # print(sorted_list)
            #print("list:", notsortedx[i])
            #print("index:", index)
            #print("count:", count)
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
            if(count % 2 == 1):
                increment += 1
            count += 1
            #itr += 1

    # print(itr)
    # print(sorted_list)
    # print(sorted_list)
    # do insertion sort on the nearly sorted list
    insertionSort(sorted_list)
    # print(sorted_list)

# end time
t1 = time.perf_counter() - t0
t1ML = time.perf_counter() - tML

print("Completely sorted by ml:", first_sorted)
print("Non sorted elements:", len(notsortedx))
print("SMLsorting:", t1ML)
print("ML sorting:", t1)

# will compare sorting times with other sorts
x_int_data = [int(math.floor(x)) for x in x_data]
x_insert = x_int_data
x_quick = x_int_data
x_bubble = x_int_data
x_heap = x_int_data
x_radix = x_int_data
x_bucket = x_int_data
# print(x_int_data)

# quick sort O(n log n)
t0 = time.perf_counter()
quickSort(x_quick, 0, total_items-1)
# print(x_quick)
t1 = time.perf_counter() - t0
print("Quick sort:", t1)

# insertion sort O(n^2)
t0 = time.perf_counter()
insertionSort(x_insert)
# print(x_insert)
t1 = time.perf_counter() - t0
print("Insertions:", t1)

# bubble sort took too long
# bubble sort O(n^2)
# t0 = time.perf_counter()
# bubbleSort(x_bubble)
# # print(x_bubble)
# t1 = time.perf_counter() - t0
# print("Bubble sort:", t1)

# heap sort O(n log n)
t0 = time.perf_counter()
heapSort(x_heap)
# print(x_heap)
t1 = time.perf_counter() - t0
print("Heap  sort:", t1)

# radix sort O(n k)
t0 = time.perf_counter()
radixSort(x_radix)
# print(x_radix)
t1 = time.perf_counter() - t0
print("radix sort:", t1)

# bucket sort O(n + k)
t0 = time.perf_counter()
bucketSort(x_bucket)
# print(x_radix)
t1 = time.perf_counter() - t0
print("bucketsort:", t1)

plt.show()
