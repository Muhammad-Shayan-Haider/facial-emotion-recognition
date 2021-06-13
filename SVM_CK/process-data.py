import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

pick_in= open('facedb.pickle','rb')
data= pickle.load(pick_in)
pick_in.close()

random.shuffle(data)

features=[]
labels=[]
c=0
for feature,label in data:
    if label < 7:
        features.append(feature)
        labels.append(label)

xtrain,xtest,ytrain,ytest= train_test_split(features,labels,test_size=0.15)
print(xtrain,ytrain)
print('successfully done data splitting')

model= SVC(C=10,kernel='rbf',gamma='auto')
for i in range(5):
    model.fit(xtrain,ytrain)
    prediction= model.predict(xtest)
    accuracy=model.score(xtest,ytest)
    print('this model is accurate upto:',accuracy)
    print('epoch ',i,'done')


pick = open('facedb.sav','wb')
pickle.dump(model,pick)
pick.close()
test_data=[xtest,ytest]
pick = open('facedb-test.pickle','wb')
pickle.dump(test_data,pick)
pick.close()
