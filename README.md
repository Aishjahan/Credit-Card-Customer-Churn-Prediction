# Customer Churn Prediction using Deep Learning

This project implements a deep learning model to predict customer churn using a dataset containing information about credit card customers. The model utilizes TensorFlow/Keras for building a neural network and demonstrates a workflow from data preprocessing to model evaluation.

## Dataset
The dataset used for this project is **Churn_Modelling.csv**, which contains customer details like:
- **CreditScore**: Customer's credit score
- **Geography**: Country of the customer
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **Balance**: Customer's account balance
- **NumOfProducts**: Number of products the customer uses
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No)
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Target variable (1 = Customer churned, 0 = Customer retained)

## Workflow
1. **Exploratory Data Analysis (EDA)**
   - Loaded the dataset using pandas.
   - Examined the structure, data types, and descriptive statistics.
   - Checked for null values and data balance.

2. **Data Preprocessing**
   - Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
   - Encoded categorical variables (`Geography`, `Gender`) using one-hot encoding.
   - Scaled numerical features using `StandardScaler`.

3. **Feature Engineering**
   - Split the dataset into features (`X`) and target (`y`).
   - Created training and testing datasets with an 80-20 split.

4. **Deep Learning Model**
   - Built a sequential neural network with:
     - Input layer with 11 features.
     - Two hidden layers with ReLU activation.
     - Output layer with a sigmoid activation for binary classification.
   - Compiled the model using `binary_crossentropy` loss and `adam` optimizer.

5. **Model Training**
   - Trained the model on the preprocessed training data.
   - Evaluated the model using accuracy and other metrics.

6. **Evaluation**
   - Evaluated the model's performance on the test data.
   - Interpreted predictions and visualized results.

## Model Summary

## Technologies Used
- **Libraries**: TensorFlow, Keras, Pandas, NumPy, Scikit-learn
- **Programming Language**: Python
- **Development Environment**: Kaggle Notebook


# Project documentation

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Aishjahan/Credit-Card-Customer-Churn-Prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:
 ```bash
   pip install -r requirements.txt
 ```
  
3. Run the notebook:
 ```bash
   jupyter notebook churn_prediction.ipynb
 ```

## **Results**

The model achieved a high level of accuracy in predicting customer churn, enabling better decision-making for customer retention strategies.


