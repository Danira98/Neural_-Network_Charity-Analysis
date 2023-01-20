# Neural Network Charity Analysis: ***Working with Deep Machine Learning Models***

## Overview of Project

### Overview
In this project, we are asked to apply our knowledge of Deep Machine Learning modules to help the company of Alphabet Soup. 
Alphabet Soup is a philanthropic foundation whose goal is to help organizations that protect the environment, improve people's well being, and unify the world. They contribute to these organizations financially, and they are interested to see the impact of their donations as well as the use of it. This analysis will help the company determine which organizations are using the foundation's money effectively and investigate how potential recipients would benefit from their donations.

### Purpose

The purpose of this project is to determine which organizations are worth investing on and predict which ones will be high risk. We are instructed to use Neural Network models, as well as an improved version of this model, to accurately predict our outcomes to at least 75%. We will achieve this by using the programming language of Python, specifically with the  help of the libraries of TensorFlow,Scikit-Learn and Pandas. By accurately predicting high risk organizations, we are able to help the company minimize monetary loses and suggest types of organizations that will be worth investing on.

## Results

###  Original Neural Network:
 
 Our Neural Network model is broken into two components, Data Preprocessing and Compiling,Training and Evaluating the Model. We will take a look at what each step entails and how it affects our model.
 
 #### Data Preprocessing:
In this step, we look at the data that has been provided to see the variables we are going to consider to be targets, features and which ones should be removed. The following list provides the breakdown of the variables in our data and in which category they fall into:
- Targets:
  - APPLICATION_TYPE
  - CLASSIFICATION
  - SPECIAL_CONSIDERATIN
  - STATUS
  - ASK_AMT
  - INCOME_AMT
  - AFFILIATION
  - USE_CASE
  - 
- Features:
   -IS_SUCCESFUL
- Neither Targets nor Features:
  - EIN
  - NAME
