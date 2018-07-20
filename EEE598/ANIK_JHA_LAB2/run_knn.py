from knn import KNN
import numpy as np
import datasets
import matplotlib.pyplot as plt

# load data
x_train, y_train, x_test, y_test = datasets.gaussian_dataset(n_train=500, n_test=500)
y = []
for i in range(1,51,5):
    model = KNN(k=i)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    #print("knn accuracy: " + str(np.mean(y_pred == y_test)))
    y.append(100.0*np.mean(y_pred == y_test))

plt.figure()

plt.plot(range(1,51,5), y)
plt.xlabel('K')
plt.ylabel('Test Accuracy(%)')
plt.title('Accuracy(%) vs K')
plt.legend()
plt.show()