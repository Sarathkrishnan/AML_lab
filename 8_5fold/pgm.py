# Python program to perform kfold cross-validation test on breast cancer dataset
#Importing required libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
#Loading the dataset
data = load_breast_cancer(as_frame = True)
df = data.frame
# Segregating dependent and independent features
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
#Implementing k-fold cross-validation
k = 5
k_fold = KFold(n_splits = k, random_state = None)
Lr = LogisticRegression(solver ="liblinear")
acc_scores = []
pre_scores = []
rec_scores = []
f1_scores = []
# Looping over each split to get the accuracy score of each split
for training_index, testing_index in k_fold.split(X):
	X_train, X_test = X.iloc[training_index,:], X.iloc[testing_index,:]
	Y_train, Y_test = Y.iloc[training_index] , Y.iloc[testing_index]
	# Fitting training data to the model
	Lr.fit(X_train,Y_train)
	# Predicting values for the testing dataset
	Y_pred = Lr.predict(X_test)
	# Calculatinf accuracy score using in-built sklearn accuracy_score method
	acc = accuracy_score(Y_pred , Y_test)
	pre = precision_score(Y_pred , Y_test)
	rec = recall_score(Y_pred , Y_test)
	f1 = f1_score(Y_pred , Y_test)
	acc_scores.append(acc)
	pre_scores.append(pre)
	rec_scores.append(rec)
	f1_scores.append(f1)
# Calculating mean accuracy score
mean_acc_score = sum(acc_scores) / k
mean_pre_score = sum(pre_scores) / k
mean_rec_score = sum(rec_scores) / k
mean_f1_score = sum(f1_scores) / k
print("Accuracy score of each fold: ", acc_scores)
print("Mean accuracy score: ", mean_acc_score)
print("\nPrecision score of each fold: ", pre_scores)
print("Mean precision score: ", mean_pre_score)
print("\nRecall score of each fold: ", rec_scores)
print("Mean recall score: ", mean_rec_score)
print("\nF-Score of each fold: ", f1_scores)
print("Mean F-Score: ", mean_f1_score)
