import numpy as np
from sklearn.svm import SVC
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
kernel = 'poly'
print "kernel: " + kernel
# 'linear', 'poly', 'rbf', 'sigmoid'
decision_function_shape = 'ovr'
# 'ovr', 'ovo'
print "decision_function_shape: " + decision_function_shape
reg = SVC(kernel=kernel, decision_function_shape=decision_function_shape)
reg.fit(x_train, y_train)
time1 = time.time()
print "--- %s seconds ---" % (time1 - start_time)

print "testing..."
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - time1)