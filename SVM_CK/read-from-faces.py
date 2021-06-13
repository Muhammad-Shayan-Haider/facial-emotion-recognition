import os
import numpy as np
import cv2
import pickle
import scipy
from PIL import Image

dir='./BestDataSet'

categories=['neutral','joy','sadness','surprise','anger','disgust','fear','open','closed','kiss','left side','right side','neutral sagital left','neutral sagital right']
data=[]
path= os.path.join(dir,'facesdb')

for subdir in os.listdir(path):
    imgDir= os.path.join(path,subdir)
    imgDir=os.path.join(imgDir,'bmp')
    for img in os.listdir(imgDir):
        imgPath= os.path.join(imgDir,img)

        em_img= Image.open(imgPath).convert('L')
        newName=img[0:-4]+'.png'
        print(newName)
        imgPath= os.path.join(imgDir,newName)
        em_img.save(imgPath,'png')
        em_img = cv2.imread(imgPath,0)
        try:
            em_img=cv2.resize(em_img,(50,50))
            image= np.array(em_img).flatten()
            data.append([image,int(img[-10:-8])])
            print(image)
        except Exception as e:
            pass
pick_in= open('facedb.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
