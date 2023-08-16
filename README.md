# Credit-Card-Fraud-Detection
## Task:
Build a Machine Learning Model to predict whether the transaction is legit or fraudulent. We have been provided with a labelled dataset, so this will come under **Supervised Learning**.</br>
This problem comes under the category of Binary Classification and we are going to build a <b>Logistic Regression Model</b> to achieve the result. 
## Dataset:
The dataset is being taken from Kaggle's **Credit Card Fraud Detection**. The dataset contains 284,807 records and 31 features. The features are:</br>
1. Time: Number of seconds elapsed between this transaction and the first transaction in the dataset</br>
2. V1, V2, V3,......., V28: may be the result of a PCA Dimensionality reduction to protect user identities and sensitive features(v1-v28)</br>
3. Amount: Transaction amount</br>
4. Class: 1 = Fraudulent Transactions, 0 = Legit Transaction</br>
</br>

Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Steps Involved:
### Import Dependencies
Libraries needed for the task are:</br>
. numpy</br>
. pandas</br>
. seaborn</br>
. train_test_split from sklearn.model_selection</br>
. LogiticRegression from sklearn.linear_model</br>
. accuracy_score from sklearn.metrics</br>
### Data Collection 
Use the read_csv function of pandas to import the dataset.</br>
shape, head, descibe and isnull() functions can be used to know our dataset.</br>
### Data Processing
The Data provided is highly Unbalanced i.e. we have 284315 data in the legit category compared to only 492 data in the fraudulent one. When feeding this kind of data to the model it might not results in a valid prediction. To overcome this we follow the following steps:</br>
1. Make separate datasets for both Normal and fraudulent transactions.</br>
2. We will perform a process called **Under Sampling** in this we will randomly choose 492 rows from the Normal Transaction using the sample function and store that to a new variable.</br>
3. Then we will concatenate the newly created legit sample with our fraud data. Thus, Our new data set contains an equal number of fraud(1) and legit(0) data.</br>
Finally, Split the dataset into features(X) and target(Y) using the **drop** function, such as the target contains only the Class column which shows whether the transaction is legit or fraudulent and the feature contains all the rest of the columns.</br>
### Split into train and test
In this, we had split the dataset into 4 parts x train, x test, y train and y test where y contains all the label data whereas x contains the features dataset.</br>
we have taken 20% of the data for training while the remaining 80% is kept for training purposes.
### Model Training
We have used Logistic Regression Algorithm to build the model for the binary classification problem.
### Model Evaluation
We have performed the model evaluation on both the training and test data. The results are:</br>
Accuracy score for trained data: 0.9453621346886912</br>
Accuracy score for test data: 0.9289340101522843
</br></br></br>

Reference: Project 10. Credit Card Fraud Detection using Machine Learning in Python | Machine Learning Projects, Siddhardhan, https://www.youtube.com/watch?v=NCgjcHLFNDg&list=PLfFghEzKVmjvuSA67LszN1dZ-Dd_pkus6&index=11

