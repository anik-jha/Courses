import numpy as np
from layers.dataset import cifar100
import matplotlib.pyplot as plt

# Please make sure that cifar-100-python is present in the same folder as dataset.py

(x_train, y_train), (x_test, y_test) = cifar100(1212356299)

from layers import (FullLayer , ReluLayer , SoftMaxLayer ,CrossEntropyLayer , Sequential)
model = Sequential(layers =( FullLayer ( 3072 , 500) , ReluLayer(), FullLayer (500,4) , SoftMaxLayer() ) , loss=CrossEntropyLayer())

lr_accuracies = np.zeros((3,))


loss1 = model.fit(x_train, y_train, lr = 0.01, epochs=15)
y_predict = model.predict(x_test)

count = 0
for i in range(np.size(y_test)):
    if y_predict[i] == y_test[i]:
        count += 1

lr_accuracies[0] = (100.0*count)/np.shape(y_predict)[0]


loss2 = model.fit(x_train, y_train, lr = 0.1, epochs=15)

y_predict = model.predict(x_test)

count = 0
for i in range(np.size(y_test)):
    if y_predict[i] == y_test[i]:
        count += 1

lr_accuracies[1] = (100.0*count)/np.shape(y_predict)[0]

loss3 = model.fit(x_train, y_train, lr = 10, epochs=15)

y_predict = model.predict(x_test)

count = 0
for i in range(np.size(y_test)):
    if y_predict[i] == y_test[i]:
        count += 1

lr_accuracies[2] = (100.0*count)/np.shape(y_predict)[0]

plt.figure(1)

plt.plot(range(1,16), loss1, label='Loss for lr = 0.01')
plt.plot(range(1,16), loss2, label='Loss for lr = 0.1')
#plt.plot(range(1,16), loss3)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss comparisons for different learning rate")
#plt.title("Loss for lr = 10")
plt.legend()
plt.show()


lr = ["0.01", "0.1", "10"]

plt.figure(2)

#plt.plot(lr,"ro", lr_accuracies, label='Accuracies')
plt.plot([0, lr_accuracies[0], lr_accuracies[1], lr_accuracies[2], 0], "ro")
plt.xticks(range(5), ["0", "0.01", "0.1", "10", "100"])
plt.xlabel("Learning rates")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy comparison for different learning rate")
plt.legend()
plt.show()
