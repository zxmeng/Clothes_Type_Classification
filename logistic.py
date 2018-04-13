import numpy as np
from sklearn.linear_model import LogisticRegression
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
penalty = 'l2'
print "penalty: " + penalty
# multi_class = 'ovr'
multi_class = 'multinomial'
print "multi_class: " + multi_class
solver = 'saga'

time1 = time.time()
reg = LogisticRegression(penalty=penalty, multi_class=multi_class, solver=solver)
reg.fit(x_train, y_train)
print "--- %s seconds ---" % (time.time() - time1)

coef = reg.coef_
coef = np.array(coef)
print "sparsity: " + str(1.0 - np.count_nonzero(coef)/float(coef.size))

print "testing..."
time2 = time.time()
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - time2)
print "--- %s seconds ---" % (time.time() - start_time)