import numpy as np
from sklearn.svm import LinearSVC
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
penalty = 'l2'
# l1
print "penalty: " + penalty
loss = 'squared_hinge'
# hinge
print "loss: " + loss

reg = LinearSVC(penalty=penalty, loss=loss)
reg.fit(x_train, y_train)

print "testing..."
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - start_time)