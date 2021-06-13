import os
import numpy as np
import cv2
import pickle

dir='./CK+48'

categories=['anger','contempt','disgust','fear','happy','sadness','surprise']
data=[]
for category in categories:
    path= os.path.join(dir,category)
    label=categories.index(category)

    for img in os.listdir(path):
        imgPath= os.path.join(path,img)
        em_img = cv2.imread(imgPath,cv2.COLOR_BGR2GRAY)
        try:
            em_img=cv2.resize(em_img,(50,50))
            image= np.array(em_img).flatten()
            data.append([image,label])
        except Exception as e:
            pass

pick_in= open('50-gray.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
