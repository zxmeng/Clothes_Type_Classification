import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
n_neighbors = 5
print "n_neighbors: " + str(n_neighbors)
# weights = 'uniform'
weights = 'distance'
print "weights: " + weights
p = 1
# 1 for manhattan_distance, 2 for euclidean_distance, other for minkowski_distance 
print "p(distance type): " + str(p)
reg = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
reg.fit(x_train, y_train)
time1 = time.time()
print "--- %s seconds ---" % (time1 - start_time)

print "testing..."
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - time1)