# Customer Churn Prediction - ANN Model

## 📋 Project Overview
This project focuses on predicting customer churn using the **churn_modelling.csv** dataset from Kaggle. The dataset contains information about customers of a bank, including their demographics, account details, and whether they have churned or not. The goal of this project is to develop an **Artificial Neural Network (ANN)** model to accurately predict customer churn and provide insights for improving customer retention strategies.

- **Objective**: Predict whether a customer is likely to churn based on their attributes and transaction history using machine learning techniques.
- **Dataset**: The `churn_modelling.csv` dataset includes features such as customer ID, gender, age, balance, number of products, and whether the customer exited the bank.

## 🔍 Data Exploration and Preprocessing
Before modeling, data exploration and preprocessing were conducted to ensure data quality and relevance:

- **Data Overview**: A thorough analysis of the dataset to understand feature distributions and relationships.
- **Missing Values**: Checked for and handled any missing values or inconsistencies in the data.
- **Feature Selection**: Identified and selected relevant features that contribute significantly to churn prediction.
- **Encoding Categorical Variables**: Categorical variables were converted into numerical format using techniques such as one-hot encoding.

## 🛠️ Model Development
The core of this project involves building an ANN model for churn prediction:

- **Model Architecture**: Designed a feedforward ANN with input, hidden, and output layers.
- **Activation Functions**: Implemented appropriate activation functions (e.g., ReLU for hidden layers, Sigmoid for output layer) to enhance learning capabilities.
- **Compilation**: Compiled the model using an optimizer (e.g., Adam) and a suitable loss function (e.g., binary cross-entropy) to evaluate model performance.
- **Training**: Trained the model on the training dataset, monitoring the performance metrics such as accuracy and loss.

## 📊 Results and Evaluation
After training the model, the performance was evaluated using various metrics:

- **Accuracy**: Measured the model's accuracy in predicting churn and non-churn customers.
- **Confusion Matrix**: Analyzed the confusion matrix to understand the model's classification performance and identify false positives/negatives.
- **ROC-AUC Score**: Calculated the ROC-AUC score to assess the model's ability to distinguish between churn and non-churn customers.

### Key Insights
- **Customer Segmentation**: Identified segments of customers at a higher risk of churn, which can inform targeted retention strategies.
- **Feature Importance**: Evaluated the influence of different features on churn prediction, providing valuable insights for business decisions.

## 💡 Future Work
Potential next steps to expand this project include:

- **Hyperparameter Tuning**: Optimize model performance by fine-tuning hyperparameters.
- **Model Comparison**: Compare the ANN model with other machine learning algorithms (e.g., Random Forest, SVM) for churn prediction.
- **Deployment**: Explore options for deploying the model as a web application or API for real-time predictions.
