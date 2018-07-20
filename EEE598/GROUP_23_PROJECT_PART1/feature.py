#####################################################################################################################
###################Import the necessary modules##################################################################

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as model
from util import plot_confusion_matrix
dtype = torch.cuda.FloatTensor
from sklearn import svm
import pickle


resize = transforms.Resize((224,224))
normalize = transforms.Normalize(mean = [0.485,0.456,0.406] ,std =[0.229 , 0.224 , 0.225])

preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize,])

#####################################################################################################################
###################Function for trainloader and testloader from dataset##############################################

def household(seed):
    np.random.seed(seed)

    preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize, ])

    hh_data_train = torchvision.datasets.ImageFolder('../data/household/train', transform=preprocessor)

    hh_data_train.train_labels =[]
    for i in range(hh_data_train.__len__()):
        hh_data_train.train_labels.append(hh_data_train.imgs[i][1])
    train_loader = torch.utils.data.DataLoader(hh_data_train,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory= True)
    #print data_train.

    hh_data_test = torchvision.datasets.ImageFolder('../data/household/test', transform=preprocessor)
    hh_data_test.test_labels = []
    for i in range(hh_data_test.__len__()):
        hh_data_test.test_labels.append(hh_data_test.imgs[i][1])
    #print hh_data_test.test_labels
    test_loader = torch.utils.data.DataLoader(hh_data_test,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory= True)  # args.nThreads)
    return train_loader, test_loader



#####################################################################################################################
###################Calling the VGG16/Resnet18 models#################################################################

def VGG_call():
    model_vgg = model.vgg16(pretrained=True).eval()
    updated_classifier = nn.Sequential(*list(model_vgg.classifier.children())[:-1])
    model_vgg.classifier = updated_classifier
    return model_vgg

def Resnet_call():
    model_res18 = model.resnet18(pretrained=True).eval()
    updated_classifier = nn.Sequential(*list(model_res18.children())[:-1])
    model_res18 = updated_classifier
    return model_res18

#####################################################################################################################
###################Fitting the SVM model using features from VGG16/Resnet18 models###################################

def SVM_fit(model1,trainloader, testloader,size,flag):
    # get data

    #print testloader.dataset
    clf = svm.SVC(decision_function_shape='ovo')


    features = torch.Tensor(len(trainloader.dataset), size)
    targets = torch.Tensor(len(trainloader.dataset))

    for i, (in_data, target) in enumerate(trainloader):
        print i

        from_ = int(i *16 )
        to_ = int(from_ + 16)

        input_var = Variable(in_data, volatile = True)
        feats_train = model1(input_var)

        features[from_:to_] = feats_train.data.cpu()
        targets[from_:to_] = target



    features= features.numpy()
    targets = targets.numpy()
    '''
    print "in_data_type",features
    print "in_data_size", np.shape(features)
    print "in_data", targets
    print "target_type", np.shape(targets)
    '''

    clf.fit(features, targets) 

    y_pred = []
    test_label = []

    for i, (in_data, target) in enumerate(testloader):
        print "test=", i

        input_var = Variable(in_data, volatile = True)
        feats_test = model1(input_var)
        if flag == 1:
            feats_test = feats_test.view(16,512)
            title = "VGG16"
        else: title = "Resnet 18"
        y_pred += list(clf.predict(feats_test.data.numpy()))
        test_label += list(target.numpy())


    #test_labels = testloader.dataset.test_labels


    print "y_pred = ", y_pred
    print "test_label", test_label
    plt.figure()
    plot_confusion_matrix(y_pred, test_label, title)

    counting2 =0

    for k in range(np.shape(y_pred)[0]):
         #print "Y_pred[i]:",ytest_predict[k]
         #print "Y_test[i]:",ytest2[i]

         if y_pred[k] == test_label[k]:
            counting2 = counting2 + 1


    testLoss = 100.0 *counting2/np.shape(y_pred)[0]
    #trainLoss = 100.0 *counting1/np.shape(y_pred_train)[0]
    print testLoss #, trainLoss
    return clf

#####################################################################################################################
###################Main Execution####################################################################################

trainloader, testloader = household(1337)

vgg_model = VGG_call()
size1 = 4096
VGG_SVM = SVM_fit(vgg_model,trainloader, testloader,size1,0)
pickle.dump(VGG_SVM, open('VGG16_SVM.pkl', 'wb'))
#Accuracy 92.9347826087%

resnet_model = Resnet_call()
size2 = 512
Resnet_SVM = SVM_fit(resnet_model,trainloader, testloader,size2,1)
pickle.dump(Resnet_SVM, open('Resnet18_SVM.pkl', 'wb'))
#Accuracy 93.4782608696


plt.show()
