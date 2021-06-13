import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

pick_in= open('facedb.pickle','rb')
data= pickle.load(pick_in)
pick_in.close()

random.shuffle(data)

features=[]
labels=[]
for feature,label in data:
    if label < 7:
        features.append(feature)
        labels.append(label)

pick_in= open('data1.pickle','rb')
data= pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
categories=['neutral','joy','sadness','surprise','anger','disgust','fear']
prev=['anger','contempt','disgust','fear','joy','sadness','surprise']

for feature,label in data:
    if label < 7:
        if prev[label] in categories:
            features.append(feature)
            labels.append(categories.index(prev[label]))

#xtrain,xtest,ytrain,ytest= train_test_split(features,labels,train_size=0.25)
#print('successfully done data splitting')
clf = MLPClassifier(activation='relu',solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(1,2500), random_state=1)
clf.fit(features,labels)

pick = open('facedb-pml-double.sav','wb')
pickle.dump(clf,pick)
pick.close()
