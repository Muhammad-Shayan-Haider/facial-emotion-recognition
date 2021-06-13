# facial-emotion-recognition
Facial expression recognition done using SVM, CNN and MLP

**Dataset Details**

Two datasets have been used, CK+ dataset and fer2013 dataset, which was used in kaggle contest [2]. Extended Cohn-Kanade Dataset (CK+) [6] has 593 sequence images of size 640 x 490. The images are grayscale. The expression labels are neutral, sadness, surprise, happiness, fear, anger, contempt and disgust. 123 subjects were used for images. 
Another dataset we used was fer2013 dataset. This is the most common dataset for facial expression recognition [7]. The original dataset is available on Kaggle. The dataset is open source and has 35,887 images, each image of 48 x 48 pixels having face images and these are all labelled as:
0: -4593 images- Angry
1: -547 images- Disgust
2: -5121 images- Fear
3: -8989 images- Happy
4: -6077 images- Sad
5: -4002 images- Surprise
6: -6198 images- Neutral

**CNN Classifier Details**

Convolutional neural network is a machine learning model used mostly for visuals. Our convolutional neural network is a five layer neural network. Some pre-processing is done before passing data to initial input layer. In the input layer, it takes image of size 48 x 48 x 1 (1 = label).

**SVM Classifier Details**

SVM is most commonly used supervised learning technique to perform classification. The model was trained after converting images into HOG features.

**MLP Classifier Details**

Multi-layered perceptron is a classical machine learning technique which is used to perform classification. To train the model there are many activation functions i.e. Sigmoid, Relu, TanH. 

**Poster of the Project**

![image](https://user-images.githubusercontent.com/60185211/121813884-8d683480-cc87-11eb-8158-5448b00d7ade.png)
