#https://datascience.stackexchange.com/questions/36049/how-to-adjust-the-hyperparameters-of-mlp-classifier-to-get-more-perfect-performa
#http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
#https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594
from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

import tensorflow as tf
import random as rn

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.callbacks import Callback, ModelCheckpoint
from keras.regularizers import l2
import csv
import p1b2

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import seaborn as sns
# Printing complete marix / full numpy array
import sys
np.set_printoptions(threshold=sys.maxsize)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#Writing in Excel
import xlwt
from xlwt import Workbook
BATCH_SIZE = 1024#64#1806111
NB_EPOCH = 20               # number of training epochs
PENALTY = 0.0001             # L2 regularization penalty
ACTIVATION = 'relu'
FEATURE_SUBSAMPLE = None
DROP = None

L1 = 16
#L2 = 8
#L3 = 4
#L4 = 8
LAYERS = [L1] #[L1,L2,L3]

class BestLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.best_val_loss = np.Inf
        self.best_val_acc = -np.Inf
        self.best_model = None

    def on_epoch_end(self, batch, logs={}):
        if float(logs.get('val_loss', 0)) < self.best_val_loss:
            self.best_model = self.model
        self.best_val_loss = min(float(logs.get('val_loss', 0)), self.best_val_loss)
        self.best_val_acc = max(float(logs.get('val_acc', 0)), self.best_val_acc)


def extension_from_parameters():
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(ACTIVATION)
    ext += '.B={}'.format(BATCH_SIZE)
    ext += '.D={}'.format(DROP)
    ext += '.E={}'.format(NB_EPOCH)
    if FEATURE_SUBSAMPLE:
        ext += '.F={}'.format(FEATURE_SUBSAMPLE)
    for i, n in enumerate(LAYERS):
        if n:
            ext += '.L{}={}'.format(i+1, n)
    ext += '.P={}'.format(PENALTY)
    return ext

def test():
    (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)
    #dist = pd.DataFrame(X_train)
    print(y_test.shape)

def recordSoftmaxProbabilities(X_train = None, y_train = None, X_test = None, y_test = None, DeterministicResults = False,fileName=None):
    if(DeterministicResults):
        __setSession()
    if(X_train is None):
        (X_train, y_train), (X_test, y_test) = p1b2.load_data()
    wb = Workbook()

    # =====create sheet1 and add headers====
    sheetToRecordTrainValidTestLossAndAccuracy = wb.add_sheet('Sheet 1')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 0, 'ValidationLoss')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 1, 'TestLoss')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 2, 'Accuracy')

    for x in range(1, 26):
        if X_train is None:
            (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]



        model = Sequential()
        model.add(Dense(LAYERS[0], input_dim=input_dim,
                        activation="sigmoid",
                        kernel_regularizer=l2(PENALTY),
                        activity_regularizer=l2(PENALTY)))

        for layer in LAYERS[1:]:
            if layer:
                if DROP:
                    model.add(Dropout(DROP))
                model.add(Dense(layer, activation=ACTIVATION,
                                kernel_regularizer=l2(PENALTY),
                                activity_regularizer=l2(PENALTY)))

        model.add(Dense(output_dim, activation='softmax'))

        #Next the model would be compiled. Compiling the model takes two parameters: optimizer and loss
        #https: // towardsdatascience.com / building - a - deep - learning - model - using - keras - 1548ca149d37
        #https://towardsdatascience.com/sequence-models-by-andrew-ng-11-lessons-learned-c62fb1d3485b
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


        print("Model Summary:", model.summary())

        ext = extension_from_parameters()
        checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)
        history = BestLossHistory()

        trainingResults = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NB_EPOCH,
                  validation_split=0.2,
                  callbacks=[history, checkpointer])

        y_pred = history.best_model.predict(X_test)
        predictedOutputs = model.predict_classes(X_test)

        scores = p1b2.evaluate(y_pred, y_test)

        #Confusion Matrix
        #cnf_matrix = confusion_matrix(y_test_SingleColumn, predictedOutputs)
        #print("Confusion Matrix = ", cnf_matrix)

        #ROC curve
        # keep probabilities for the positive outcome only
        #ns_probs = [0 for _ in range(len(y_test_SingleColumn))]
        #lr_probs = y_pred[:, 0]
        #print("Faqeer = ", lr_probs)
        # calculate scores
        #ns_auc = roc_auc_score(y_test_SingleColumn, ns_probs)
        #lr_auc = roc_auc_score(y_test_SingleColumn, lr_probs)
        #print('No Skill: ROC AUC=%.3f' % (ns_auc))
        #print('Logistic: ROC AUC=%.3f' % (lr_auc))

        #Print Other Results
        testResults = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE)
        print('Evaluation on test data:', scores)
        #print('Test Scores [Test Loss, Test Accuracy] = ', testResults[0])
        #print('Loss: ', np.amin(trainingResults.history['loss']),'Accuracy: ',np.amin(trainingResults.history['accuracy']),'Val_Loss: ',np.amin(trainingResults.history['val_loss']),'Val_Accuracy :',np.amin(trainingResults.history['val_accuracy']))
        #print('best_val_loss={:.5f} best_val_acc={:.5f}'.format(history.best_val_loss, history.best_val_acc))
        #print('Best model saved to: {}'.format('model'+ext+'.h5'))

        # ======Save Training loss,Validation(Best model) loss, test loss and Accuracy
        #sheetToRecordTrainValidTestLossAndAccuracy.write(x, 0, str(round(np.amin(trainingResults.history['loss']), 3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 0, str(round(history.best_val_loss, 3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 1, str(round(testResults[0], 3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 2, str(scores))
        # ===========================================================================
        # =====Save Instance level outputs against each experiment/iteration over for Each Class=====
        # =====create sheet2 and add headers====
        sheetToRecordInstanceLevelOutput = wb.add_sheet('IterationNo' + str(x))
        sheetToRecordInstanceLevelOutput.write(1, 0, 'InputFeatures')
        sheetToRecordInstanceLevelOutput.write(1, 1, 'Expected_OR_ActualOutput')
        sheetToRecordInstanceLevelOutput.write(1, 2, 'PredictedOutput')
        sheetToRecordInstanceLevelOutput.write(1, 3, 'Probabilities')
        sheetToRecordInstanceLevelOutput.write(1, 4, 'MaxProbability')
        startRowToBeInserted = 2
        for x in range(X_test.shape[0]):
            # print("ddd = ", X_test[x])
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 0, 'Test Data Input Features')  # str(X_test[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 1, str(y_test[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 2, str(predictedOutputs[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 3, str(y_pred[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 4, str(np.amax(y_pred[x])))
            startRowToBeInserted = startRowToBeInserted + 1
        # ==============================================================================

        submission = {'scores': scores,
                      'model': model.summary(),
                      'submitter': 'Developer Name' }

    if fileName != None:
        wb.save(fileName)  # .xls
    else:
        wb.save("Default.xls")  # .xls
    # print('Submitting to leaderboard...')
    # leaderboard.submit(submission)
    __resetSeed()
    # return history.best_model
    return scores


#https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
#https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223
def UnivariateSelection():
    data = pd.read_csv("C:/Users/faqeerrehman/MSU/Research/CancerPrediction/ScientificSWTesting/Data/Pilot1/P1B2.train.csv")
    #data = pd.read_csv("C:/Users/faqeerrehman/MSU/Research/CancerPrediction/ScientificSWTesting/Data/Pilot1/Train.csv")
    X = data.iloc[:, 2:]  # independent columns
    y = data.iloc[:, 1]  # target column i.e price range
    #print("X = ", X)
    #print("Y = ", y)
    # apply SelectKBest class to extract top 10 best features
    bestfeatures = SelectKBest(score_func=chi2, k=3000)

    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    #print(dfcolumns.to_string())
    bestFeaturesWithScores = featureScores.nlargest(3000, 'Score')
    print(bestFeaturesWithScores.to_string())  # print 10 best features
    #print(bestFeaturesWithScores)
    #print(extractFeaturesIndexFromFile())

def extractFeaturesIndexFromFile():
    concatenatedColumnValues = ''
    data = pd.read_csv("C:/Users/faqeerrehman/MSU/Research/CancerPrediction/ScientificSWTesting/Data/Pilot1/BestFeaturesResult.csv")
    for index, row in data.iterrows():
        firstIndexColumnValues = row[0].split(' ')
        concatenatedColumnValues = concatenatedColumnValues + ',' + str(int(firstIndexColumnValues[0])+2)
    print(concatenatedColumnValues[1:])
    #return concatenatedColumnValues[1:]
        #for i in range(length):
            #print(i)


def featureImportance():
    data = pd.read_csv("C:/Users/faqeerrehman/MSU/Research/CancerPrediction/ScientificSWTesting/Data/Pilot1/P1B2.train.csv")
    X = data.iloc[:, 2:]  # independent columns
    y = data.iloc[:, 1]  # target column i.e price range
    #print("X = ", X)
    #print("Y = ", y)
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

def correlationMatrixwithHeatmap():
    data = pd.read_csv("C:/Users/faqeerrehman/MSU/Research/CancerPrediction/ScientificSWTesting/Data/Pilot1/P1B2.train.csv")
    X = data.iloc[:, 2:]  # independent columns
    y = data.iloc[:, 1]  # target column i.e price range
    # get correlations of each features in dataset
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")

def mainFeatureSelection(X_train = None, y_train = None, X_test = None, y_test = None, DeterministicResults = False):
    if(DeterministicResults):
        __setSession()



    if X_train is None:
        (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    print("X Train: ", X_train)
    print("Y Train: ", y_train)
    model = Sequential()
    model.add(Dense(LAYERS[0], input_dim=input_dim,
                    activation=ACTIVATION,
                    kernel_regularizer=l2(PENALTY),
                    activity_regularizer=l2(PENALTY)))

    for layer in LAYERS[1:]:
        if layer:
            if DROP:
                model.add(Dropout(DROP))
            model.add(Dense(layer, activation=ACTIVATION,
                            kernel_regularizer=l2(PENALTY),
                            activity_regularizer=l2(PENALTY)))

    model.add(Dense(output_dim, activation=ACTIVATION))


    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())

    ext = extension_from_parameters()
    checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)
    history = BestLossHistory()

    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=NB_EPOCH,
              validation_split=0.2,
              callbacks=[history, checkpointer])

    y_pred = history.best_model.predict(X_test)
    predictedOutputs = model.predict_classes(X_test)
    #print("TestDataX = ", X_test)
    #print("TestDataY = ", y_test)
    #i=0
    #j=1

    #for x in np.nditer(X_test, flags = ['external_loop'], order = 'C'):
     #   print("Instance = ", X_test[i:j], " --> Prediciton = ", history.best_model.predict(np.array(X_test[i:j])))
     #   i = i + 1
      #  j = j + 1
        #print("Loop Iterations : ", x)

    #print("Y_Pred = " , y_pred)
    #print("PredictedOutputs = ", predictedOutputs)
    scores = p1b2.evaluate(y_pred, y_test)
    print('Evaluation on test data:', scores)

    print('best_val_loss={:.5f} best_val_acc={:.5f}'.format(history.best_val_loss, history.best_val_acc))
    print('Best model saved to: {}'.format('model'+ext+'.h5'))



    submission = {'scores': scores,
                  'model': model.summary(),
                  'submitter': 'Developer Name' }

    # print('Submitting to leaderboard...')
    # leaderboard.submit(submission)
    __resetSeed()
    #return history.best_model
    return scores

def mainFaqeer(X_train = None, y_train = None, X_test = None, y_test = None, DeterministicResults = False,fileName = ""):
    if(DeterministicResults):
        __setSession()

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')

    for x in range(1,3):
        print("Run-----> ", x)
        if X_train is None:
            (X_train, y_train), (X_test, y_test) = p1b2.load_data(n_cols=FEATURE_SUBSAMPLE)

        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        model = Sequential()

        model.add(Dense(LAYERS[0], input_dim=input_dim,
                        activation=ACTIVATION,
                        kernel_regularizer=l2(PENALTY),
                        activity_regularizer=l2(PENALTY)))

        for layer in LAYERS[1:]:
            if layer:
                if DROP:
                    model.add(Dropout(DROP))
                model.add(Dense(layer, activation=ACTIVATION,
                                kernel_regularizer=l2(PENALTY),
                                activity_regularizer=l2(PENALTY)))

        model.add(Dense(output_dim, activation=ACTIVATION))


        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())

        ext = extension_from_parameters()
        checkpointer = ModelCheckpoint(filepath='model'+ext+'.h5', save_best_only=True)
        history = BestLossHistory()

        model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NB_EPOCH,
                  validation_split=0.2,
                  callbacks=[history, checkpointer])

        y_pred = history.best_model.predict(X_test)

        print('best_val_loss={:.5f} best_val_acc={:.5f}'.format(history.best_val_loss, history.best_val_acc))
        print('Best model saved to: {}'.format('model'+ext+'.h5'))

        scores = p1b2.evaluate(y_pred, y_test)
        print('Evaluation on test data:', scores)
        #sheet1.write(x, 0, str(scores))
        sheet1.write(x, 0, str(np.amax(y_pred)))
        sheet1.write(x, 1, str(scores))
        submission = {'scores': scores,
                      'model': model.summary(),
                      'submitter': 'Developer Name' }

        # print('Submitting to leaderboard...')
        # leaderboard.submit(submission)
    wb.save(fileName)
    __resetSeed()
    return history.best_model

def __resetSeed():
    np.random.seed()
    rn.seed()

def __setSession():
    # Sets session for deterministic results
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development


    # The below is necessary in Python 3.2.3 onwards to
    # have reproducible behavior for certain hash-based operations.
    # See these references for further details:
    # https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
    # https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
    import os
    os.environ['PYTHONHASHSEED'] = '0'

    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(42)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    # Force TensorFlow to use single thread.
    # Multiple threads are a potential source of
    # non-reproducible results.
    # For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    from keras import backend as K
    # The below tf.set_random_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    # tf.global_variables_initializer()
    tf.compat.v1.set_random_seed(1234)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

    # Fixed by Faqeer ur Rehman on 24 Nov 2019
    #K.set_session(sess)
    tf.compat.v1.keras.backend.set_session(sess)


if __name__ == '__main__':
    #mainToRecordTrainValidateTestLosses()
    recordSoftmaxProbabilities(None,None,None,None,DeterministicResults = False, fileName= "SourceOrg.xls")
