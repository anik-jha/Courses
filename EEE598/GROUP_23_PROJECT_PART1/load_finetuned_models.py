import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as model
from util import plot_confusion_matrix
dtype = torch.cuda.FloatTensor

use_gpu = torch.cuda.is_available()
resize = transforms.Resize((224,224))
normalize = transforms.Normalize(mean = [0.485,0.456,0.406] ,std =[0.229 , 0.224 , 0.225])

preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize,])


def household(seed):
    np.random.seed(seed)

    preprocessor = transforms.Compose([resize, transforms.ToTensor(), normalize, ])

    hh_data_train = torchvision.datasets.ImageFolder('../data/household/train', transform=preprocessor)

    hh_data_train.train_labels =[]
    for i in range(hh_data_train.__len__()):
        hh_data_train.train_labels.append(hh_data_train.imgs[i][1])
    #print hh_data_train.train_labels
    train_loader = torch.utils.data.DataLoader(hh_data_train,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=1,
                                              pin_memory= True)
    #print data_train.

    hh_data_test = torchvision.datasets.ImageFolder('../data/household/test', transform=preprocessor    )
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




def train_model(model, criterion, optimizer,train_loader, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0] * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects*1.0 / dataset_sizes

        print('Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, test_loader,title):
    # switch to evaluate mode
    #self.eval()

    running_corrects = 0
    y_pred =[]
    y_true = []

    for i, data in enumerate(test_loader, 0):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data)
        y_pred +=list(preds)
        y_true +=list(labels.data)

    test_acc = running_corrects * 1.0 / test_dataset_sizes

    print('Test_Acc: {:.4f}'.format(test_acc))
    plt.figure()
    plot_confusion_matrix(y_pred, y_true, title)

    return y_pred


def VGG_call():
    ####################Main function
    model_vgg = model.vgg16(pretrained=True)
    mod = list(model_vgg.classifier.children())
    mod.pop()
    mod.append(nn.Linear(4096,9))
    new_classifier = nn.Sequential(*mod)
    model_vgg.classifier = new_classifier
    return model_vgg

def Resnet_call():

    model_resnet = model.resnet18(pretrained=True)
    model_resnet.fc = nn.Linear(512, 9)
    return model_resnet


def finetune(best_model, testloader,title):



    #criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    y_pred = test_model(best_model, testloader,title)
    return best_model


# get data
trainloader, testloader = household(1337)

dataset_sizes = np.shape(trainloader.dataset.train_labels)[0]
test_dataset_sizes = np.shape(testloader.dataset.test_labels)[0]

#vgg_model = VGG_call()

vgg_model = torch.load('VGG16_finetuned.pt')
title1 = 'VGG16_Finetune'
vgg_best = finetune(vgg_model, testloader, title1)
#torch.save(vgg_best, 'VGG16_finetuned.pt')
'''
#resnet_model = Resnet_call()
resnet_model = torch.load('Resnet18_finetuned.pt')
title2 = 'Resnet18_Finetune'
resnet_best = finetune(resnet_model, testloader,title2)
#torch.save(resnet_model, 'Resnet18_finetuned.pt')
'''


plt.show()