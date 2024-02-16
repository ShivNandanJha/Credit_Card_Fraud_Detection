# Credit Card Fraud Detection using Machine Learning

This project is aimed at detecting fraudulent credit card transactions using machine learning techniques. We utilize Jupyter Notebook as the primary development environment for our code.

## Overview

Credit card fraud is a significant issue for both financial institutions and cardholders. Detecting fraudulent transactions is crucial for preventing financial losses and maintaining trust in the banking system. In this project, we employ machine learning algorithms to classify transactions as either legitimate or fraudulent based on various features.

## Dataset

We use the Credit Card Fraud Detection dataset from Kaggle, which contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly imbalanced, with only 492 fraud cases out of 284,807 transactions. Each transaction contains numerical input features that are the result of a PCA transformation due to privacy concerns.

Dataset Link: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Jupyter Notebook

The Jupyter Notebook `credit_card_fraud_detection.ipynb` contains the code for data preprocessing, model training, evaluation, and visualization. It walks through the following steps:

1. Data loading and exploration
2. Data preprocessing (e.g., scaling, handling imbalance)
3. Model selection and training (e.g., Logistic Regression, Random Forest, etc.)
4. Model evaluation (e.g., accuracy, precision, recall, F1-score)
5. Visualizations of model performance

## Requirements

To run the Jupyter Notebook, you need the following dependencies:

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install these dependencies using pip:

```
pip install jupyter numpy pandas scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

```
git clone https://github.com/yourusername/credit-card-fraud-detection.git
```

2. Navigate to the project directory:

```
cd credit-card-fraud-detection
```

3. Launch Jupyter Notebook:

```
jupyter notebook
```

4. Open `credit_card_fraud_detection.ipynb` in your browser and execute the cells sequentially to reproduce the results
