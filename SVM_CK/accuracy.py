import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import sys



categories=['anger','contempt','disgust','fear','happy','sadness','surprise']

pick_in= open('50model.sav','rb')
model= pickle.load(pick_in)
pick_in.close()

pick_in= open('50-gray-test.pickle','rb')
xtest,ytest= pickle.load(pick_in)
pick_in.close()
prediction=model.predict(xtest)
accuracy=model.score(xtest,ytest)
print('Accuracy :',accuracy)
print('Prediction:',prediction,ytest)
