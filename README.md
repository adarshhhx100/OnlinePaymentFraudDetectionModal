# Online-Payment-Fraud-Detection-using-Machine-Learning-in-Python

# Payment Fraud Detection

This project aims to detect fraudulent online payments using machine learning techniques. The provided dataset contains information about various online transactions, and the goal is to build a model that can accurately classify transactions as fraudulent or genuine.

## Steps

1. **Data Exploration and Visualization:**
   - Load the dataset and examine its structure and features.
   - Visualize the distribution of different transaction types and amounts.
   - Analyze the correlation between features using a heatmap.

2. **Data Preprocessing:**
   - Encode categorical variables (e.g., transaction type) using one-hot encoding.
   - Drop irrelevant columns (e.g., 'nameOrig', 'nameDest').
   - Split the dataset into training and testing sets.

3. **Model Training:**
   - Experiment with different machine learning models, including:
     - Logistic Regression
     - XGBoost Classifier
     - Support Vector Machine (SVM)
     - Random Forest Classifier
   - Impute missing values using a SimpleImputer.
   - Train the models on the training data.

4. **Model Evaluation:**
   - Evaluate the performance of each model using the ROC AUC score.
   - Visualize the confusion matrix for the best-performing model.

## Requirements

- Python 3.x
- Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost

## How to Run

1. Install the required libraries:
2. Place the dataset file ('onlinefraud.csv') in the same directory as the code.
3. Run the Python code in a Jupyter Notebook or a suitable Python environment.

## Results

The project demonstrates the application of machine learning for fraud detection. The best-performing model can be selected based on the ROC AUC score and further optimized for deployment in a real-world setting.
