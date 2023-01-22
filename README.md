# Neural Network Charity Analysis: ***Working with Deep Machine Learning Models***

## Overview of Project

### Overview
In this project, we are asked to apply our knowledge of Deep Machine Learning modules to help the company of Alphabet Soup. 
Alphabet Soup is a philanthropic foundation whose goal is to help organizations that protect the environment, improve people's well being, and unify the world. They contribute to these organizations financially, and they are interested to see the impact of their donations as well as the use of it in these companies. This analysis will help the company determine which organizations are using the foundation's money efficiently and investigate how potential recipients would benefit from their donations.

### Purpose

The purpose of this project is to determine which organizations are worth investing on and predict which ones will be high risk. We are instructed to use Neural Network models, as well as an improved version of this model, to obtain an accuracy of at least 75% if possible. We will achieve this by using the programming language of Python, specifically with the help of the libraries of TensorFlow,Scikit-Learn and Pandas. By accurately predicting high risk organizations, we are able to help the company minimize monetary loses and suggest types of organizations that will be worth investing on.

## Results
 
Our Neural Network model is broken into two components, Data Preprocessing and Compiling,Training and Evaluating the Model. We will take a look at what each step entails and how it affected our model.
 
### Data Preprocessing:
In this step, we look at the data that has been provided to see the variables we are going to consider to be targets, features and which ones should be removed. The following list provides the breakdown of the variables in our data and in which category they fall into:

- Features:
  - APPLICATION_TYPE
  - SPECIAL_CONSIDERATION
  - STATUS
  - ASK_AMT
  - INCOME_AMT
- Target:
  - IS_SUCCESSFUL
- Other Variables:
  - AFFILIATION
  - CLASSIFICATION
  - USE_CASE
  - EIN
  - NAME
  
We chose our target to be the column "IS_SUCCESSFUL" since the purpose of our algorithm is to identify whether the organization has been efficiently using the money provided. We assign the target to be represented as y, in which all the values of said column are passed as an array to later be trained and tested.

![data_preprocessing](https://user-images.githubusercontent.com/111034667/213823435-2d62b095-cb20-478e-a702-b7dea8316907.png)

Through the process of data processing, we encode our features with the method One Hot Encoding in which we split our columns into unique values and bin unique values of lower amount into a category of "Other". Additionally, during this process we drop the variables under the Other variables category since they are not necessary for our model and it allows us to process our model faster and obtain a smaller dataset to work with. We chose the features to be represented as X, in which all the values of the columns are passed as an array.

![encoded_features](https://user-images.githubusercontent.com/111034667/213823174-730a3a55-0710-43e2-9fef-3c15cc332c49.png)

![data_preprocessing](https://user-images.githubusercontent.com/111034667/213823435-2d62b095-cb20-478e-a702-b7dea8316907.png)

Lastly, we take our y and X arrays, and split them into training and testing datasets to use into our model. We additionally create two variables, X_train_scaled and X_test_scaled, to scale our train and tested data.

![training_set](https://user-images.githubusercontent.com/111034667/213823357-ed358435-be74-4f5f-84f3-70ce66c523f3.png)


### Compiling, Training, and Evaluating the Model

During this process, we pick a deep neural network model to compile, train and evaluate our data. In order to attempt to achieve our desired goal, we begin by creating a model and testing its accuracy. If this model does not reach the desired accuracy, we move on to adjusting our model to ideally improve our model and get our accuracy closer to the desired percentage.

#### Model 1

Our first model contained the following characteristics:

- Layers and hidden nodes: 
  - First layer: contains 80 nodes, and uses the activation function of Relu.
  - Second layer: contains 30 nodes, and uses the activation function of Relu.
  - Output layer: contains one node, and uses the activation function of Sigmoid.

![model1_layer](https://user-images.githubusercontent.com/111034667/213825196-03382fbe-9063-4c05-b8e8-832f7d3b9b4f.png)

- Epochs:

We assigned our model to train with our fit function in which the X_train and y_train variables are trained with 100 epochs. Additionally, it uses the call back function which saves the weights of every 5 epoch into an HDF5 file called AlphabetSoupCharity.
 
![model1_epoch](https://user-images.githubusercontent.com/111034667/213825094-f075bed4-be4f-4818-871f-09196bbd885c.png)

After running our code, this model obtained an accuracy of 0.6815. Since the accuracy is not above 75%, we proceeded to optimize this model in an attempt to increase the accuracy score.

#### Optimization Model 1

In an attempt to optimize our model, we dropped the columns 'EIN','NAME','ORGANIZATION' and 'CLASSIFICATION' of our original data at the beginning of the process of data processing before creating our bins.

![data_preprocessing_att1](https://user-images.githubusercontent.com/111034667/213828254-8d39b4da-c3e5-41c3-9dee-bb56faf39d5d.png)

Additionally, we changed the way we binned our APPLICATION_TYPE values. We increased our condition to bin every type of application that contained less than 700 values to be under the category of "Other". 

![app_type_bin](https://user-images.githubusercontent.com/111034667/213828273-bf143c20-46ac-4f42-bbbc-634b8f40915e.png)

We kept the rest of the code from Model 1, and after running our model, we observed that the accuracy score had increase to 0.7135.

![accuracy_opt_1](https://user-images.githubusercontent.com/111034667/213828379-839635fe-7cc5-462f-98f8-21ba79b1ac27.png)

Note: This model ended up obtaining the closest accuracy score to 75%, therefore, we saved the weights of every 5 epochs into the HDF5 file of AlphabetSoupCharity_Optimization.

#### Optimization Model 2

We continued to work on our model to ideally improve our accuracy. This time, we kept the processed data from Optimization Model 1 but changed the amount of layers and increase the epoch number. 

- Layers:
  - First : contains 80 nodes, and uses the activation function of Relu.
  - Second : contains 30 nodes, and uses the activation function of Relu.
  - Third: contains 50 nodes, and uses the activation function of Relu.
  - Output: contains one nodes, and uses the activation functino of sigmoid.

![model2_layer](https://user-images.githubusercontent.com/111034667/213903198-59fcfb6e-a223-4aaa-8b05-39b793318836.png)

- Epoch:

The amount of epoch in our fit_model line is now increased to 110.

![model2_epoch](https://user-images.githubusercontent.com/111034667/213903216-3a29f8fd-cf62-4bf7-ae0e-8c9c4bc6759b.png)

After running our code for this version of optimized model, we observed the accuracy score to decrease to 0.5449

![accuracy_opt_2](https://user-images.githubusercontent.com/111034667/213903250-dc12f503-1b91-40df-8cb0-3d55124f308b.png)

#### Optimization Model 3

For our last attempt, we continued to build upon Model 1 but this time we added two layes with different activation functions, increase the amount of neurons and change the epoch number.

- Layers:
  - First : contains 100 nodes, and uses the activation function of Relu.
  - Second : contains 80 nodes, and uses the activation function of Relu.
  - Third: contains 50 nodes, and uses the activation function of Relu.
  - Fourth: contains 30 nodes, and uses the activation function of sigmoid.
  - Output: contains one nodes, and uses the activation functino of sigmoid.
 
![model3_layer](https://user-images.githubusercontent.com/111034667/213903824-9023fbb2-3710-40af-bf38-581e1c4d6023.png)

- Epoch:

The amount of epoch in our fit_model line is now decreased to 80.

![model3_epoch](https://user-images.githubusercontent.com/111034667/213903834-16a9851e-9847-460b-a257-39adf5c44c39.png)

With our modifications, the optimized model decreased the accuracy scored to 0.4675.

![accuracy_opt_3](https://user-images.githubusercontent.com/111034667/213903854-d222456c-6184-48c1-aa56-db476d95fefd.png)

## Summary

Overall, our attempts were not able to meet the accuracy score desired of at least 75%. As we have observed from the results section, changing the way we binned our data did improve our model but changing the epoch, amount of layers, amount of nodes, and activation codes decreased the accuracy score. This shows us that neural networks might not be the ideal model to use for this type of data, due to the fact that neural networks are known to have a problem of overfitting our data. To achieve the desired accuracy score, we can attempt to take another route in which we now use the model of Logistic Regression. Logistic Regression is known to be a classification algorithm, and by implementing this model, we will be able to categorize the types of organizations that will effectively use the donations at a faster rate. With the proper data preprocessing, specifically at the point of binning our data together, we will be able to achieve a higher accuracy score with the Logistic Regression model.
