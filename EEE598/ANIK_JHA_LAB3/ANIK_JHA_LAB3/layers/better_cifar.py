import numpy as np
from layers.dataset import cifar100

# Please make sure that cifar-100-python is present in the same folder as dataset.py

(x_train, y_train), (x_test, y_test) = cifar100(1212356299)

from layers import (FullLayer , ReluLayer , SoftMaxLayer ,CrossEntropyLayer , Sequential_better)
model = Sequential_better(layers =( FullLayer ( 3072 , 1500) , ReluLayer(),FullLayer (1500 , 500) , ReluLayer(), FullLayer (500,4) , SoftMaxLayer() ) , loss=CrossEntropyLayer())

loss2 = model.fit(x_train, y_train, lr = 0.1, epochs=15)
y_predict = model.predict(x_test)

count = 0
for i in range(np.size(y_test)):
    if y_predict[i] == y_test[i]:
        count += 1

accuracy = (100.0*count)/np.shape(y_predict)[0]

print "Accuracy of better CIFAR = ", accuracy, "%"

