import numpy as np
from sklearn.kernel_ridge import KernelRidge
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
kernel = 'linear'
print "kernel: " + kernel
reg = KernelRidge(alpha=1, kernel=kernel)
reg.fit(x_train, y_train)
print reg.alpha_

print "testing..."
print "accuracy: " +  str(reg.score(x_test, y_test))
print "--- %s seconds ---" % (time.time() - start_time)