# ANN Classification and Regression Project

This project demonstrates the implementation of Artificial Neural Networks (ANN) for both *classification* and *regression* tasks using *TensorFlow/Keras, along with **hyperparameter tuning* and *Streamlit deployment*.

## Project Overview

This repository contains:
- *ANN Classification* on the Churn Modelling dataset
- *ANN Regression* implementation
- *Hyperparameter tuning* for ANN using Keras Tuner
- *Streamlit web app* for customer churn prediction

## Files in this Repository

- ann.py – ANN model implementation
- app.py – Streamlit app for customer churn prediction
- experiments.ipynb – ANN classification experiments and model training
- prediction.ipynb – Prediction notebook for churn classification
- regression.ipynb – ANN regression practical implementation
- hyperparametertuningann.ipynb – Hyperparameter tuning for ANN
- requirements.txt – Required Python libraries
- Churn_Modelling.csv – Dataset used for classification

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Keras Tuner

## ANN Classification Task

The classification part of the project predicts whether a customer is likely to churn based on features such as:
- Geography
- Gender
- Age
- Balance
- Credit Score
- Estimated Salary
- Number of Products
- Tenure
- Credit Card Status
- Active Member Status

### Workflow
1. Load and preprocess the dataset
2. Encode categorical variables
3. Scale numerical features
4. Train ANN classification model
5. Predict churn probability
6. Deploy with Streamlit

## ANN Regression Task

The regression notebook demonstrates how ANN can be used for a regression problem using TensorFlow/Keras.

### Regression workflow
1. Load dataset
2. Preprocess features
3. Build ANN regression model
4. Train the model
5. Evaluate performance

## Hyperparameter Tuning

Hyperparameter tuning is implemented using *Keras Tuner* to find the best ANN architecture.

Tuned parameters include:
- Number of input units
- Number of hidden layers
- Number of neurons in hidden layers
- Learning rate

## Streamlit App

The project includes a Streamlit web application for customer churn prediction.

To run the app locally:

```bash
streamlit run app.py