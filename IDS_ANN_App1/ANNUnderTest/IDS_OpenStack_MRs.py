# Metamorphic testing
# Relations and their explanations may be found at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3019603/

import IDS_OpenStack as p1b2_baseline_keras2
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
        (self.X_train, self.y_train), (self.X_test, self.y_test) = p1b2.load_data()

    def test_MRs(self):
        # Original
        p1b2_baseline_keras2.recordSoftmaxProbabilities(None, None, None, None, DeterministicResults=False,
                                                        fileName="Source.xls")
        ## BEGIN MR0

        print("MR0 Executing")
        (X_train, y_train), (X_test, y_test) = self.__getCopiesOfData()
        numFeatures = X_train.shape[1]

        # Makes an array of random size filled with random numbers between 0 and numFeatures without repitition
        randomSubset = np.random.choice(range(numFeatures), np.random.randint(numFeatures), False)

        #k = np.random.randint(1, 100)
        #b = np.random.randint(100)

        ###Very Small change / scaling / shifting
        k = 1;  # np.random.randint(1, 100)
        b = 10;  # np.random.randint(100)
        for i in range(X_train.shape[1]):
            X_train[:, i] = X_train[:, i] * k + b
            X_test[:, i] = X_test[:, i] * k + b

        p1b2_baseline_keras2.recordSoftmaxProbabilities(X_train, y_train, X_test, y_test, False,"MR0.xls")
        return
        ## END MR0



    def __shuffleColumns(self, x):
        x = np.transpose(x)
        np.random.shuffle(x)
        x = np.transpose(x)

    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    def __shuffleColumnsInUnison(self, a, b, ):
        a = np.transpose(a)
        b = np.transpose(b)

        assert len(a) == len(b)
        p = np.random.permutation(len(a))

        a = np.transpose(a[p])
        b = np.transpose(b[p])
        return a, b

    def __getCopiesOfData(self):
        return (copy.copy(self.X_train), copy.copy(self.y_train)), (copy.copy(self.X_test), copy.copy(self.y_test))

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