import numpy as np
from sklearn.mixture import GaussianMixture
import time
import process

start_time = time.time()
print "loading data..."
x_train, y_train, x_test, y_test = process.read_data()

print "training..."
n_components = 5
print "n_components: " + str(n_components)
covariance_type = 'spherical'
print "covariance_type: " + covariance_type
reg = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
reg.fit(x_train, y_train)
print reg.alpha_

print "testing..."
print "accuracy: " +  str(1.0 - np.count_nonzero(y_test - reg.predict(x_test)) / float(len(y_test)))
print "--- %s seconds ---" % (time.time() - start_time)
