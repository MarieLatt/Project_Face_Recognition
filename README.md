# Face recognition of the 6 characters from Friends

The goal of this project is to build a program that takes a picture containing one or several characters from the Friends TV show and returns, as accurately as possible, their names (Monica, Rachel, Phoebe, Ross, Chandler, Joey).

This project contains 3 directories for the 3 parts of the project.
Parts 1 and 2 correspond two different approaches for the classification of character faces.

1. training_NN: building and training a custom convolutional Neural Network
2. training_SVM: using a trained Neural Network ([facenet](https://github.com/timesler/facenet-pytorch)) to generate a vector representation of a face and then training a classification algorithm (SVM)
3. demo: pipeline code that takes as input a picture and returns the predicted names from the 2 models

## Model 1: training_NN

This directory contains:
1. scraping/IMDB_scraping.ipynb: web scraping of all the pictures related to the Friends TV show on IMDb
2. cropping/FaceCropping.ipynb: detection and cropping faces out of the pictures with MTCNN from [facenet](https://github.com/timesler/facenet-pytorch).  
These 2 steps allow building the dataset to train the convolutional Neural Network.
3. model_nn.py: the Neural Network model (consisting in 4 convolutional layers and 3 linear layers)
4. NN_train_test.ipynb: training the Neural Network and validation

## Model 2: training_SVM

- embedding_and_classification.ipynb: using [MTCNN and InceptionResnetV1 from facenet](https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb) for face detection, cropping and conversion to a vector representation. It uses a dataset of 20 pictures of each character in separate directories organized as follows:

photos_friends_actors  
├── Chandler  
├── Joey  
├── Monica  
├── Phoebe  
├── Rachel  
└── Ross
 
The obtained vectors are used as input to train and test a SVM algorithm.

- data/svm_model.joblib: trained SVM model
 
## Demo

- demo_final_project.py: pipeline code that takes as input a picture and returns the predicted names from the 2 models.

## Tools

* Python
* Numpy
* Pandas
* requests
* BeautifulSoup
* Matplotlib
* Seaborn
* Scikit-Learn
* PyTorch
* OpenCV


