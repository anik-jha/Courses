
#############################Import necessary modules###########################################
from dataset import load_dataset
import torch
import torch.nn as nn
import pickle
import torch.optim as optim
import torch.autograd
from torch.autograd import Variable
import numpy as np


############################# GRU Function ###########################################
class GRU(nn.Module):

    def __init__(self, dim_embedding, dim_hidden, vocabulary_size, classes):
        super(GRU, self).__init__()
        self.dim_hidden = dim_hidden
        self.word_embeddings = nn.Embedding(vocabulary_size, dim_embedding)
        self.gru = nn.GRU(dim_embedding, dim_hidden)
        self.hidden2 = nn.Linear(dim_hidden, classes)
        self.hidden = self.initialize_hidden()

    def initialize_hidden(self):
        return Variable(torch.zeros(1, batch_size, self.dim_hidden))

    def forward(self, sentences, labels):
        embeds = self.word_embeddings(sentences)
        gru_out, self.hidden = self.gru(
            embeds.view(len(sentences), batch_size, -1), self.hidden)

        last_seq = gru_out[-1, :, :]
        hidden_op = self.hidden2(last_seq)
        return hidden_op, labels



#############################Setting the nessary hyper-parameters##########################################

torch.manual_seed(1) # set the seed for uniform result
batch_size = 128
dim_hidden = 40
dim_embedding = 70
vocabulary_size = 28049
classes = 2
epochs = 7


GRU_model = GRU(dim_embedding, dim_hidden, vocabulary_size,classes)
loss_layer= nn.CrossEntropyLoss(reduce=False)
optimizer = optim.Adam(GRU_model.parameters(), lr =0.01)
trainiterator, testiterator = load_dataset("spam", batch_size) # loading the dataset

#############################Training##########################################

count1 = 0
correct_train = 0
for i in range(epochs):
    epoch_loss = 0
    count1+=1

    for batch in trainiterator:
        GRU_model.hidden = GRU_model.initialize_hidden()
        sentence = batch.sentence
        label = batch.label
        updated_batch = sentence.size()[1]
        label_updated = label.size()[0]
        if updated_batch != batch_size or label_updated != batch_size:
            continue

        GRU_model.zero_grad()
        hidden_op,labels = GRU_model.forward(sentence,label)
        train_loss = loss_layer(hidden_op,labels)
        train_loss.sum().backward()
        optimizer.step()

        epoch_loss += train_loss.data[0]
        print("[Epoch: %d] loss: %.3f" %
				(i+1, epoch_loss/(i+1)))
        epoch_loss = 0

#############################Saving the trained model###########################################

pickle.dump(GRU_model, open('GRU_spam.pkl', 'wb'))

#############################Testing############################################################
correct =0
count_test = 0
GRU_model.eval()
for batch in testiterator:
    sentence = batch.sentence
    label = batch.label
    updated_batch = sentence.size()[1]
    label_updated = label.size()[0]
    if updated_batch != batch_size or label_updated != batch_size:
        continue
    spam_space,y_test_pred = GRU_model.forward(sentence,label)
    count_test +=1
    for i in range(batch_size):
        news_space = spam_space.data.numpy()
        class_label = np.argmax(news_space[i])
        prediction = class_label
        if prediction == y_test_pred.data[i]:
            correct += 1
        else:
            a=1
print("accuracy is: ")
print(correct,count_test*batch_size)
print(correct*100.0/(count_test*batch_size))
#94.6875
