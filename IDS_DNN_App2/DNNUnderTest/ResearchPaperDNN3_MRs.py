# Metamorphic testing
# Relations and their explanations may be found at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3019603/

import ResearchPaperDNN3 as p1b2_baseline_keras2
import p1b2 as p1b2

import keras
import numpy as np
import copy

import unittest

# Writing in Excel
import xlwt
from xlwt import Workbook


class p1b2Tests(unittest.TestCase):

    # Note: test cases only run when they start with 'test'

    @classmethod
    def setUpClass(self):
        #(self.X_train, self.y_train), (self.X_test, self.y_test), (self.X_SmallDataTest) = p1b2.load_dataDNN3()
        print("SetupClass...")

    def test_MRs(self):
        # Original
        #p1b2_baseline_keras2.recordSoftmaxProbabilities(None, None, None, None, None,None,
                                                       # fileName="Source.xls")
        #return;
        ## BEGIN MR0
        print("MR0 Executing")
        (X_train, y_train), (X_test, y_test), (X_SmallDataTest, y_SmallDataTest) = p1b2.load_dataDNN3()
        numFeatures = X_train.shape[1]

        ###Very Small change / scaling / shifting
        k = 1;  # np.random.randint(1, 100)
        b = 0.3;  # np.random.randint(100)
        for i in range(X_train.shape[1]):
            X_train[:, i] = X_train[:, i] * k + b
            X_test[:, i] = X_test[:, i] * k + b
            X_SmallDataTest[:, i] = X_SmallDataTest[:, i] * k + b

        p1b2_baseline_keras2.recordSoftmaxProbabilities(X_train, y_train, X_test, y_test, X_SmallDataTest,
                                                        y_SmallDataTest,
                                                        "MR0.xls")
        print("MR0 Completed")
        return
        ## BEGIN MR12
        print("MR12 Executing")
        (X_train, y_train), (X_test, y_test), (X_SmallDataTest, y_SmallDataTest) = p1b2.load_dataDNN3()

        X_train, X_test, X_SmallDataTest = self.__shuffleColumnsInUnison(X_train, X_test, X_SmallDataTest)

        p1b2_baseline_keras2.recordSoftmaxProbabilities(X_train, y_train, X_test, y_test, X_SmallDataTest,
                                                        y_SmallDataTest,
                                                        "MR12.xls")
        ## End MR12

        ## BEGIN MR21
        print("MR21 Executing")
        (X_train, y_train), (X_test, y_test), (X_SmallDataTest, y_SmallDataTest) = p1b2.load_dataDNN3()
        tempTrain = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
        tempTest = np.zeros((X_test.shape[0], X_test.shape[1] + 1))
        tempTestSmall = np.zeros((X_SmallDataTest.shape[0], X_SmallDataTest.shape[1] + 1))

        tempTrain[:, :-1] = X_train
        tempTest[:, :-1] = X_test
        tempTestSmall[:, :-1] = X_SmallDataTest
        # =====assigning any other uninformative value to the new added feature====
        # tempTrain[:,tempTrain.shape[1]-1] = 600;
        p1b2_baseline_keras2.recordSoftmaxProbabilities(tempTrain, y_train, tempTest, y_test, tempTestSmall,
                                                        y_SmallDataTest, "MR21.xls")

        ## End MR12

    def __shuffleColumns(self, x):
        x = np.transpose(x)
        np.random.shuffle(x)
        x = np.transpose(x)

    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def __shuffleColumnsInUnison(self, a, b, c, ):
        a = np.transpose(a)
        b = np.transpose(b)
        c = np.transpose(c)

        assert len(a) == len(b) == len(c)
        p = np.random.permutation(len(a))

        a = np.transpose(a[p])
        b = np.transpose(b[p])
        c = np.transpose(c[p])
        return a, b, c

    def __getCopiesOfData(self):
        return (copy.copy(self.X_train), copy.copy(self.y_train)), (copy.copy(self.X_test), copy.copy(self.y_test)), (copy.copy(self.X_SmallDataTest))

    def __permuteLabels(self, y_train, y_test):
        p = np.arange(y_train.shape[1])
        np.random.shuffle(p)

        for x in range(y_train.shape[0]):
            i = np.where(y_train[x, :] > .5)
            y_train[x, i] = 0
            y_train[x, p[i]] = 1

        for x in range(y_test.shape[0]):
            i = np.where(y_test[x, :] > .5)
            y_test[x, i] = 0
            y_test[x, p[i]] = 1

        return p


if __name__ == '__main__':
    unittest.main()