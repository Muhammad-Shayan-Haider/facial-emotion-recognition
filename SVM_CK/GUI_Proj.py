import PySimpleGUI as sg
import os
import pickle
import cv2
import numpy as np
from PIL import Image
import time
categories=['neutral','joy','sadness','surprise','anger','disgust','fear']
sg.theme('Dark Green 5')
layout=[[sg.Text('Welcome To facial expression Recognizer')],
[sg.Text('Browse to Image'),sg.FileBrowse()],
[sg.Button('Predict'), sg.Button('Cancel')]]

window = sg.Window('FER-AI', layout).finalize()
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks cancel
        break
    print('You entered ', values['Browse'])
    window.close()
    try:
        path=os.path.join(values['Browse'])
        em_img=cv2.imread(path)
        im=Image.open(values['Browse'])
        em_img=cv2.cvtColor(em_img,cv2.COLOR_BGR2GRAY)
        image=[]
        em_img=cv2.resize(em_img,(50,50))
        image= np.array(em_img).flatten()
        im.resize((50,50),Image.ANTIALIAS)
        im = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)

        im.save('./tmp.gif')
        sg.theme('DarkAmber')
        layout=[[sg.Text('You enter This Image')],
        [sg.Image(filename="tmp.gif", key= "IMG1",visible=True)],
        [sg.Text('Model is loading to predict',key='Prediction')]]
        window = sg.Window('FER-AI-Prediction', layout).finalize()
        pick_in= open('facedb.sav','rb')
        model= pickle.load(pick_in)
        pick_in.close()
        print('model loaded')
        print(image)
        label= model.predict([image])
        window.FindElement('Prediction').Update(categories[label[0]])
        window.FindElement('Prediction').Update(text_color='red')
        while True:
            event,values = window.read()
            if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks cancel
                os.remove('./tmp.gif')
                break
            sleep(5)
            break


        # do stuff
    except IOError:
        sg.theme('Dark Red 1')
        layout=[[sg.Text('Your File is not an image file\n')],[sg.Text('Try Again\n')]]
        window = sg.Window('FER-AI-file-Error', layout).finalize()
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:	# if user closes window or clicks cancel
                break
        break
