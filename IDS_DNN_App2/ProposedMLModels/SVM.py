


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

##WithoutMetaDataFeatures####
dataFilePath = '/DataSetWithHighAccuracy/MR0/GeneratedDataSet/WithOutMetaDataFeatures/train.csv'
##WithMetaDataFeatures####
#dataFilePath = '/DataSetWithHighAccuracy/MR0/GeneratedDataSet/WithMetaDataAndProbabilityFeatures/train.csv'


#dataFilePath = 'C:/Users/faqeerrehman/MSU/OneDrive - Montana State University/Research/Clem/ResearchPaperDNN3/Results20Epochs/Research2/DataSetWithHighAccuracy/NewDS/MR0/GeneratedDataBackup/WithIncreaseTheNoOfIterations/train_1.csv'

data = pd.read_csv(dataFilePath, engine='c')
#shuffle data
data = data.sample(frac=1, random_state=2016)
#data = shuffle(data)
####Remove duplicate observations/rows
#print("Before", data.shape)
data.drop_duplicates(keep=False,inplace=True)
#col_names = ['prob1','prob2','AL','isSaturday','AK','packets','isSYN','isFIN','AJ','isFriday','AH','dstPt','AG','AI','classLabel']
#data = data[col_names]
print("Before", data.shape)
y = data.classLabel
X = data.drop('classLabel', axis=1)
#X = np.sqrt(X)
#print(X.head())

#exit()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=213) #Use 201,211,212


from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()

# encoding train labels
encoder.fit(y_train)
Y_train = encoder.transform(y_train)

# encoding test labels
encoder.fit(y_test)
Y_test = encoder.transform(y_test)
#print(Y_train)
#print(len(Y_test))

#Total Number of Continous and Categorical features in the training set
num_cols = X_train._get_numeric_data().columns
#print("Number of numeric features:",num_cols.size)



names_of_predictors = list(X_train.columns.values)

# Scaling the Train and Test feature set
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X3_train_scaled = scaler.fit_transform(X_train)
X3_test_scaled = scaler.transform(X_test)

import pylab as pl

import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

### Parameter settings
#params_grid = [{'kernel': ['rbf'], 'gamma': [1000,100,10,1,0.1,0.01,0.001], #search range from 10-3 to 10+3
#                     'C': [1000,100,10,1,0.1,0.01,0.001],'probability':[True]}]  ##search range from 10-3 to 10+3

####Best parameter found for data without metadata features
#params_grid = [{'kernel': ['rbf'], 'gamma': [0.1], #0.1
#                     'C': [100],'probability':[True]}]

####Best parameter found for data with metadata features
params_grid = [{'kernel': ['rbf'], 'gamma': [0.01], #search range from 10-3 to 10+3
                     'C': [1000],'probability':[True]}]  ##search range from 10-3 to 10+3


print("Start");
svm_model = GridSearchCV(SVC(), params_grid, cv=10)
svm_model.fit(X3_train_scaled, Y_train)
print("Fitted");
# View the accuracy score
print('Best score for training data:', svm_model.best_score_,"\n")

# View the best parameters for the model found using grid search
print('Best C:',svm_model.best_estimator_.C,"\n")
print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")
print('Best Probability:',svm_model.best_estimator_.probability,"\n")

final_model = svm_model.best_estimator_
Y_pred = final_model.predict(X3_test_scaled)
Y_pred_label = list(encoder.inverse_transform(Y_pred))


#######Good read to understand precion, recall and F1: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9####
Y3_test_label = list(encoder.inverse_transform(Y_test))
print("Accuracy=",accuracy_score(Y3_test_label,Y_pred_label))
print("\n")
print("Confusion Matrix=",confusion_matrix(Y3_test_label,Y_pred_label))
print("\n")
print("Classification Report=",classification_report(Y3_test_label,Y_pred_label))

print("Training set score for SVM: %f" % final_model.score(X3_train_scaled , Y_train))
print("Testing  set score for SVM: %f" % final_model.score(X3_test_scaled  , Y_test ))

print("SVM Score = ",svm_model.score)

cm = confusion_matrix(Y3_test_label,Y_pred_label)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + list(np.unique(Y_pred_label)))
ax.set_yticklabels([''] + list(np.unique(Y_pred_label)))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=True)
plt.interactive(False)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
Y_df = final_model.predict_proba(X3_test_scaled)
#Multi-class
#auc1 = roc_auc_score(Y_test,Y_df,multi_class='ovr',average="macro")
#auc2 = roc_auc_score(Y_test,Y_df,multi_class='ovr',average="weighted")
#Binary-class
#auc1 = roc_auc_score(Y_test,Y_df)
#auc2 = roc_auc_score(Y_test,Y_df)
#print("AUC1 = ",auc1)
#print("AUC2 = ",auc2)


