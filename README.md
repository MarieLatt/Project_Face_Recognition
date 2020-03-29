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
3. model_nn_bn_cv4.py and model_nn_bn_cv4x2.py: the Neural Network models 
 I tested several configurations of the neural network: 
  - changing the learning rate, the optimizer, the batchsize,       
  - adding a Dropout(0.2)  or a batchnorm to the convolutional layers ([ref](https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd)): BatchNorm2d gave me slightly better results and a faster convergence,     
  - modifying the number of out-channels for the convolutional layers:  I took my inspiration from the architecture of [VGG-19](https://www.researchgate.net/figure/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means_fig2_325137356). Depths of 64,128,256 and 512 pushed my laptop to its limits, so I used 32,64, 256 and 512 filters for the convolutional layers (final model = model_nn_bn_cv4.py), and I tried to double the convolutional layers like in the first layers of VGG-19 (model_nn_bn_cv4x2.py) 
  
4. NN_train_test.ipynb: training the Neural Network and Evaluation of the models

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

## Future Improvements

* Improving the custom convolutional Neural Network would require more data for the training: more pictures or data augmentation
* Modifying the models to include an 'Other' category for the secondary characters (for example, replacing the SVM classifier by a Radius Neighbors Classifier in Model 2?)
