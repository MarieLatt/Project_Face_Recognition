# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:18:28 2019

@author: Marie
"""
## importing libraries

import cv2
import os
## for plotting
import matplotlib.pyplot as plt

## importing pytorch for NN
import torch

## importing sklearn for classification

from sklearn import svm
from sklearn.preprocessing import Normalizer

## importing joblib for loading the SVM model
from joblib import load

## importing facenet for facial recognition
# read https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
from facenet_pytorch import MTCNN, InceptionResnetV1

## variables
workers = 0 if os.name == 'nt' else 4
MIN_FACE_CONFIDENCE = 0.95
INPUT_FOLDER = 'images_to_test'
PYT_PATH = 'Conv_net_image.pyt'
svm_dict={0:'Chandler', 1:'Joey', 2:'Monica', 3:'Phoebe', 4:'Rachel', 5:'Ross'}
nn_dict={0:'Rachel', 1:'Ross', 2:'Chandler', 3:'Phoebe', 4:'Monica', 5:'Joey'}
name_list=list(svm_dict.values())


print('Please wait while I download the files')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True, keep_all=True, 
    device=device
)


## functions


def load_picture(pict_name):
    from PIL import Image
    print('Loading the picture...')
    path=INPUT_FOLDER+'\\'+pict_name
    im = Image.open(path)
    im.show()
    return im

def cropping_faces(image):
    print('Detecting and cropping faces out of the picture...')
    aligned=[]
    x_aligned, prob = mtcnn(image, return_prob=True)
    for face,p in  zip(x_aligned,prob):
        #print(f'Face detected with probability: {p:.6f}')
        if p>=MIN_FACE_CONFIDENCE:
            aligned.append(face)
    return aligned


def get_embeddings(tensors):
    print('Calculating the vectors corresponding to the face features...')
    X =torch.stack(tensors).to(device) # .permute(0,3,1,2)
    embeddings = resnet(X.detach().cpu()) 
    return embeddings

def data_preparation(tensors):
    X_vectors=tensors.data.numpy()
    in_encoder = Normalizer(norm='l2')
    X_norm = in_encoder.transform(X_vectors)
    return X_norm

def SVM_prediction(X):
    print('Predicting the names using SVM prediction')
    model=load('data_SVM\svm_model.joblib')
    y_hat=model.predict(X)
    predicted_names=[svm_dict[i] for i in y_hat]
    return predicted_names
        

def NN_prediction(cropped_images):
    print('Predicting the names using my NN')
    from model_nn import Net
    model = Net().to(device)
# load best state of the model
    model.load_state_dict(torch.load(PYT_PATH))
    X=torch.stack(cropped_images,dim=0)
    # Invert red and blue channel as the network is trained with BGR images
    X = torch.stack([X[:,2,...],X[:,1,...],X[:,0,...]],dim=1)
    X_in=X/torch.max(X)
# set to eval for inference
    model.eval()
    with torch.no_grad():
        output = model(X_in)
        Y_hat = output.argmax(dim=1, keepdim=True).numpy().squeeze()
    predicted_names=[nn_dict[i] for i in Y_hat]
    return predicted_names
    
def display_results(cropped_faces, names, model, block = False):
    l=len(cropped_faces)
    d={'NN':"Faces Recognized by my NN", 'SVM':"Faces Recognized by SVM Classification"}
    fig=plt.figure(figsize=(16,5))
    fig.suptitle(d[model], fontsize=16)
    i=1
    for face, name in zip(cropped_faces, names):
        plt.subplot(1,l,i)
        plt.imshow(face.numpy().transpose(1,2,0))
        plt.axis('off')
        plt.title(name)
        i+=1
    plt.show(block=block)

def display_2results(cropped_faces,names_NN,names_SVM):
    l=len(cropped_faces)
    fig=plt.figure(figsize=(12,5))
    grid = plt.GridSpec(11,4*l,wspace=0.4, hspace=1)
    txt1=fig.add_subplot(grid[0,:4*l])
    txt1.axis('off')
    txt1.text(0.5, 0.5, "Faces Recognized by my NN", ha='center', fontsize=18)
    txt2=fig.add_subplot(grid[6,:4*l])
    txt2.axis('off')
    txt2.text(0.5, 0.5, "Faces Recognized by SVM Classification", ha='center', fontsize=18)
    i=0
    for face, name in zip(cropped_faces, names_NN):
        fig.add_subplot(grid[1:5,4*i:4*(i+1)])
        plt.axis('off')
        plt.imshow(face.numpy().transpose(1,2,0))
        plt.title(name)
        i+=1
    i=0
    for face, name in zip(cropped_faces, names_SVM):
        fig.add_subplot(grid[7:11,4*i:4*(i+1)])
        plt.axis('off')
        plt.imshow(face.numpy().transpose(1,2,0))
        plt.title(name)
        i+=1
    plt.show(block=True)

#%%
original_image=input('Name of the file? ')  
image=load_picture(original_image)      

#%%
cropped_faces=cropping_faces(image)
embeddings=get_embeddings(cropped_faces)
#%%        

X=data_preparation(embeddings)
predicted_names_SVM=SVM_prediction(X)
#display_results(cropped_faces,predicted_names_SVM, model='SVM')

predicted_names_NN = NN_prediction(cropped_faces)

#display_results(cropped_faces,predicted_names_NN, model='NN', block = True)

display_2results(cropped_faces,predicted_names_NN,predicted_names_SVM)

 















