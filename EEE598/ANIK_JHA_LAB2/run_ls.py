"""
Run least squares with provided data
"""

import numpy as np
import matplotlib.pyplot as plt
from ls import LeastSquares
import pickle

# load data
data = pickle.load(open("ls_data.pkl", "rb"))
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']


def root_mean_square(prediction, actual):
    return (np.power((prediction - actual), 2)).mean()


# try ls
temp_train = []
temp_test = []
for i in range(1,21):
    temp_mse_train = []
    ls = LeastSquares(i)
    ls.fit(x_train, y_train)
    pred_train = ls.predict(x_train)
    pred_test = ls.predict(x_test)
    mse_train = root_mean_square(pred_train, x_train)
    temp_train.append(mse_train)
    mse_test = root_mean_square(pred_test, x_test)
    #print "     RMSE_train = ", mse_train, " RMSE_test = ", mse_test
    temp_test.append(mse_test)

plt.figure(1)

plt.plot(range(1,21), temp_train, label='Train error')
plt.plot(range(1,21), temp_test, label='Test error')
plt.xlabel("Degree of polynomial")
plt.ylabel("Error")
plt.title("Least square error plot")
plt.legend()
plt.show()


