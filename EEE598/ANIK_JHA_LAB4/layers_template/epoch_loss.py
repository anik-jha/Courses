import numpy as np
import matplotlib.pyplot as plt
from layers_template.dataset import cifar100

(xtrain, ytrain), (xtest, ytest) = cifar100(1212356299)

# print(np.shape(ytrain))

from layers_template import (ConvLayer, FullLayer, ReluLayer, MaxPoolLayer, SoftMaxLayer, FlattenLayer,
                             CrossEntropyLayer, Sequential)

# model = Sequential(layers=(FullLayer(3072,500), ReluLayer(),  FullLayer(500,128), ReluLayer(), FullLayer(128,4),
#                                                   SoftMaxLayer()),
#                                                  loss=CrossEntropyLayer())

model = Sequential(layers=(
    ConvLayer(3, 16, 3), ReluLayer(), MaxPoolLayer(2), ConvLayer(16, 32, 3), ReluLayer(), MaxPoolLayer(2),
    FlattenLayer(),
    FullLayer(2048, 4), SoftMaxLayer()),
    loss=CrossEntropyLayer())


# model = Sequential(layers=(ConvLayer(3,3,16), ReluLayer(),MaxPoolLayer(2,2), ConvLayer(3,3,32), ReluLayer(),MaxPoolLayer(2,2),FlattenLayer(),FullLayer(4,1),SoftMaxLoss()),
#                                                   loss=CrossEntropyLayer())

counting2 = 0
counting = 0
numEpoch = 10
trainLoss = np.zeros((numEpoch))
testLoss = np.zeros((numEpoch))

ytrain2 = ytrain
ytrain2 = np.argmax(ytrain2,axis=1)

ytest2 = ytest
ytest2 = np.argmax(ytest2,axis=1)

for i in range(numEpoch):

    loss = model.fit(xtrain,ytrain,1,0.1,128)
    ytrain_predict = model.predict(xtrain)
    ytest_predict = model.predict(xtest)

    counting=0
    counting2=0

    for J in range(np.shape(ytrain_predict)[0]):
        #print "Ytrain_pred[i]:",ytrain_predict[J]
        #print "Y_train[i]:",ytrain2[J]
        if ytrain_predict[J] == ytrain2[J]:
            counting= counting+ 1



    for k in range(np.shape(ytest_predict)[0]):
        #print "Y_pred[i]:",ytest_predict[k]
        #print "Y_test[i]:",ytest2[i]

        if ytest_predict[k] == ytest2[k]:
            counting2 = counting2 + 1

    trainLoss[i] = 100.0 - ((100.0*counting)/np.size(ytrain_predict))
    testLoss[i] = 100.0 - ((100.0 *counting2)/np.size(ytest_predict))


print "Accuracy Train:",trainLoss
print "Accuracy Test:",testLoss

plt.figure(1)
a= plt.plot(trainLoss,label = 'Training Loss:')
b = plt.plot(testLoss,label = 'Testing Loss:')
plt.xlabel("Epoch:")
plt.ylabel("Loss:")
plt.title("Train/Test Loss Vs Number of Epochs")
plt.show()


