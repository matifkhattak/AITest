#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#https://elitedatascience.com/machine-learning-algorithms #### Comparisons of ML algorithms

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import dataLoader
from sklearn.utils import shuffle

##WithoutMetaDataFeatures####
dataFilePath = '/DataSetWithHighAccuracy/MR0/GeneratedDataSet/WithOutMetaDataFeatures/train.csv'
##WithMetaDataFeatures####
#dataFilePath = '/DataSetWithHighAccuracy/MR0/GeneratedDataSet/WithMetaDataAndProbabilityFeatures/train.csv'

data = pd.read_csv(dataFilePath, engine='c')
##shuffle data
data = data.sample(frac=1, random_state=2016)
#data = shuffle(data)
#data['classLabel']=(data['classLabel']=='Bug').astype(int)
####Remove duplicate observations/rows
#print("Before", data.shape)
data.drop_duplicates(keep=False,inplace=True)
#noOfBuggyCodeRows = data.loc[data['classLabel'] != 'NoBug']
#print(noOfBuggyCodeRows.shape)
#exit()
#print("Before", data.shape)
#print(data.corr())


#print(data.head(10))
y = data.classLabel
X = data.drop('classLabel', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=210) #212

#import dataLoader
#(X_train, y_train), (X_test, y_test) = dataLoader.load_dataDNN3()

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



#matplotlib inline
import seaborn as sns
from sklearn.utils import shuffle
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV


print("Start");
#https://www.featureranking.com/tutorials/machine-learning-tutorials/sk-part-3-cross-validation-and-hyperparameter-tuning/#1.5
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


# Create the parameter grid based on the results of random search
###Begin For mutants having low accuracy#####
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [10,20,30,40,50,90],
#    'max_features': [7,8,9,10,11,12],
#    'min_samples_leaf': [1,2,3,4,5,6], #in %age
#    'min_samples_split': [5,6,7,8,9,10,11], #in %age
#    'n_estimators': [30,31,32,33,34,35,36,37,38,39,40,41,42,43,50,60,70]
#}

####Best parameter found for data without metadata features
#param_grid = {
#    'bootstrap': [True],
#    'max_depth': [10], ##40
#    'max_features': [23], ##11
#    'min_samples_leaf': [7], ##50
#    'min_samples_split': [7], ##8
#    'n_estimators': [50] ##33
#}
####Best parameter found for data with metadata features
param_grid = {
    'bootstrap': [True],
    'max_depth': [18],
    'max_features': [29],
    'min_samples_leaf': [5], #in %age
    'min_samples_split': [6], #in %age
    'n_estimators': [59]
}
# Create a based model
randomForestModel  = RandomForestClassifier(criterion='entropy')

gs = GridSearchCV(estimator = randomForestModel, param_grid = param_grid,
                          cv = 10, n_jobs = -1, verbose = 2)
fitted_model = gs.fit(X3_train_scaled, Y_train)
print("Fitted");
#print('Best score for training data:', fitted_model.best_score_,"\n")

final_model = fitted_model.best_estimator_

print('Best Model:', final_model )
Y_pred = final_model.predict(X3_test_scaled)
Y_pred_label = list(encoder.inverse_transform(Y_pred))


#######Good read to understand precion, recall and F1: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9####
Y3_test_label = list(encoder.inverse_transform(Y_test))


print("Accuracy=",accuracy_score(Y3_test_label,Y_pred_label))
print("\n")
print("Confusion Matrix=",confusion_matrix(Y3_test_label,Y_pred_label))
print("\n")
print("Classification Report=",classification_report(Y3_test_label,Y_pred_label))

print("Training set score for RandomForest: %f" % final_model.score(X3_train_scaled , Y_train))
print("Testing  set score for RandomForest: %f" % final_model.score(X3_test_scaled  , Y_test ))

print("RandomForest Score = ",fitted_model.score)
#####AUC
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
Y_df = final_model.predict_proba(X3_test_scaled)[:,1]

#Binary-class
auc1 = roc_auc_score(Y_test,Y_df)
auc2 = roc_auc_score(Y_test,Y_df)
print("AUC1 = ",auc1)
print("AUC2 = ",auc2)

#############Visualize Tree###################
#print(X.columns.values) print only independent features names (excluding class label)
#print(np.unique(y.values)) #print only unique values of class label
from sklearn import tree
plt.figure(figsize=(20,20))
_ = tree.plot_tree(final_model.estimators_[1], feature_names=X.columns.values, filled=True)
#plt.show()

#############End Visualize Tree###############

# PLOT ROC curve
#https://medium.com/cascade-bio-blog/making-sense-of-real-world-data-roc-curves-and-when-to-use-them-90a17e6d1db
from sklearn.metrics import roc_curve, auc
# get false and true positive rates

fpr, tpr, thresholds = roc_curve(Y_test, Y_df, pos_label=1) #pos_label means label of positive class, we can also remove this parameter
# get area under the curve
roc_auc = auc(fpr,tpr)
plt.figure(dpi=150)
plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
plt.title('ROC Curve for RF classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend()
#plt.show()
#####End AUC
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
#plt.show(block=True)
plt.interactive(True) #False

###############Begin Featur,.e Importance given by Random Fores###########################
#https://stackoverflow.com/questions/44101458/random-forest-feature-importance-chart-using-python
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1] #np.argsort(importances)
#for name, importance in zip(data.columns, final_model.feature_importances_):
#    print(name, "=", importance)
    # Print the feature ranking
print("Feature ranking:")
for f in range(X3_train_scaled.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, data.columns[indices[f]], importances[indices[f]]))
#plot them
features = data.columns

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show(block=True)
###############End feature importance given by Random Fores#############################