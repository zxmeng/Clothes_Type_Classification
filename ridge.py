import numpy as np
from sklearn.linear_model import RidgeCV
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
time1 = time.time()
reg = RidgeCV(alphas=[15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
reg.fit(x_train, y_train)
print "--- %s seconds ---" % (time.time() - time1)
print reg.alpha_

coef = reg.coef_
print "sparsity: " + str(1.0 - np.count_nonzero(coef)/float(len(coef)))

print "testing..."
time2 = time.time()
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - time2)
print "--- %s seconds ---" % (time.time() - start_time)