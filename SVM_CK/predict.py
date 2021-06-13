import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import sys

path = input("Enter the path of your file: ")



categories=['anger','contempt','disgust','fear','happy','sadness','surprise']

pick_in= open('facedb-pml.sav','rb')
model= pickle.load(pick_in)
pick_in.close()
em_img=cv2.imread(path)
em_img=cv2.cvtColor(em_img,cv2.COLOR_BGR2GRAY)
image=[]
try:
    em_img=cv2.resize(em_img,(50,50))
    image= np.array(em_img).flatten()
except Exception as e:
    pass
label= model.predict([image])
print(categories,label)
