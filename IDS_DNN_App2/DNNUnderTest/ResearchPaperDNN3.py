#Evaluating Shallow and Deep Neural Networks for Network Intrusion Detection Systems in Cyber Security
#https://ieeexplore.ieee.org/document/8494096
#https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/blob/master/Evaluating%20Shallow%20and%20Deep%20Neural%20Networks%20for%20Network%20Intrusion%20Detection%20Systems%20in%20Cyber%20Security.pdf
#https://github.com/rahulvigneswaran/Intrusion-Detection-Systems/blob/master/dnn/dnn4test.py

from __future__ import print_function
#from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
#np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import p1b2
from xlwt import Workbook
from datetime import datetime

def recordSoftmaxProbabilities(X_train = None, y_train = None, X_test = None, y_test = None, X_SmallDataTest = None,y_SmallDataTest = None, fileName=None):



    if(X_train is None):
        (X_train, y_train), (X_test, y_test), (X_SmallDataTest, y_SmallDataTest) = p1b2.load_dataDNN3()
    batch_size = 2048;#64
    input_dim = X_train.shape[1]
    no_of_epochs = 20#20#20
    no_Of_Iterations = 21#31
    wb = Workbook()
    # =====create sheet1 and add headers====
    sheetToRecordTrainValidTestLossAndAccuracy = wb.add_sheet('Sheet 1')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 0, 'Accuracy')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 1, 'TimeTaken(InMinutes)')
    sheetToRecordTrainValidTestLossAndAccuracy.write(0, 2, 'TimeTaken(InSeconds)')
    #sheetToRecordTrainValidTestLossAndAccuracy.write(0, 3, 'F1score')

    for x in range(1, no_Of_Iterations):
        start = datetime.now()

        print("Run =====> ", x)
        # 1. define the network
        model = Sequential()
        model.add(Dense(1024,input_dim=input_dim,activation='relu')) #input_dim=41
        model.add(Dropout(0.01))
        model.add(Dense(768,activation='relu')) #768
        model.add(Dropout(0.01))
        model.add(Dense(512,activation='relu'))
        model.add(Dropout(0.01))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))

        # try using different optimizers and different optimizer configs

        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        ##Begin Commented By Faqeer ur Rehman##
        #checkpointer = callbacks.ModelCheckpoint(filepath="kddresults/dnn4layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
        #csv_logger = CSVLogger('kddresults/dnn4layer/training_set_dnnanalysis.csv',separator=',', append=False)
        #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=2, callbacks=[checkpointer,csv_logger])
        #End Commented by Faqeer ur Rehman#
        #checkpointer = callbacks.ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True, monitor='loss')
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=no_of_epochs) #, callbacks=[checkpointer]
        #model.save("kddresults/dnn4layer/dnn4layer_model.hdf5")
        #pred = model.predict_classes(X_test)
        #proba = model.predict_proba(X_test)
        #np.savetxt("dnnres/dnn4predicted.txt", pred)
        #np.savetxt("dnnres/dnn4probability.txt", proba)

        pred = model.predict_classes(X_test)
        #Custom code for small test dataset results
        predSmallData = model.predict_classes(X_SmallDataTest)
        y_pred = model.predict(X_test)
        y_predSmallDataset = model.predict(X_SmallDataTest)
        print("Model Summary:", model.summary())
        #=======Record Accuracy, recall, precision, f1======#

        #accuracy = accuracy_score(y_test, pred)
        #recall = recall_score(y_test, pred , average="binary")
        #precision = precision_score(y_test, pred , average="binary")
        #f1 = f1_score(y_test, pred, average="binary")

        scores = p1b2.evaluateAccuracy(y_pred, y_test)
        ##Calcualate time taken by each iteration(in minutes, seconds)
        end = datetime.now()
        timeDifference = end - start
        hours, remainder = divmod(timeDifference.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        ####################################################3

        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 0, str(round(scores, 3)))
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 1, minutes)
        sheetToRecordTrainValidTestLossAndAccuracy.write(x, 2, seconds)
        print("Scores ==> ", str(round(scores, 3)))
        #print("time_inhoursminutesSeconds =", minutes)
        #print("time_inhoursminutesSeconds =", seconds)
        #return
        #sheetToRecordTrainValidTestLossAndAccuracy.write(x, 1, str(round(precision, 3)))
        #sheetToRecordTrainValidTestLossAndAccuracy.write(x, 2, str(round(recall, 3)))
        #sheetToRecordTrainValidTestLossAndAccuracy.write(x, 3, str(round(f1, 3)))

        # =======End Record Accuracy, recall, precision, f1======#

        # =====Save Instance level outputs against each experiment/iteration over for Each Class=====
        # =====create sheet2 and add headers====
        sheetToRecordInstanceLevelOutput = wb.add_sheet('IterationNo' + str(x))
        sheetToRecordInstanceLevelOutput.write(0, 0, 'InputFeatures')
        sheetToRecordInstanceLevelOutput.write(0, 1, 'Expected_OR_ActualOutput')
        sheetToRecordInstanceLevelOutput.write(0, 2, 'PredictedOutput')
        sheetToRecordInstanceLevelOutput.write(0, 3, 'Probabilities')
        sheetToRecordInstanceLevelOutput.write(0, 4, 'MaxProbability')
        startRowToBeInserted = 1
        for x in range(X_SmallDataTest.shape[0]):
            # print("ddd = ", X_test[x])
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 0,
                                                   'Test Data Input Features')  # str(X_test[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 1, str(y_SmallDataTest[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 2, str(predSmallData[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 3, str(y_predSmallDataset[x]))
            sheetToRecordInstanceLevelOutput.write(startRowToBeInserted, 4, str(np.amax(y_predSmallDataset[x])))
            startRowToBeInserted = startRowToBeInserted + 1


        #print("----------------------------------------------")
        #print("accuracy")
        #print("%.3f" %accuracy)
        #print("precision")
        #print("%.3f" %precision)
        #print("recall")
        #print("%.3f" %recall)
        #print("f1score")
        #print("%.3f" %f1)

    if fileName != None:
        wb.save(fileName)  # .xls
    else:
        wb.save("Default.xls")  # .xls
    return ""
if __name__ == '__main__':

    #mainToRecordTrainValidateTestLosses()
    recordSoftmaxProbabilities(fileName= "SourceDNN3.xls")
