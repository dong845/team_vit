import numpy as np


class ConfusionMatrix:
    def __init__(self, pred, actual, width, height, classes):
        self.pred = pred
        self.actual = actual
        self.width = width
        self.height = height
        self.classes = classes

    def construct(self):

        # -------------converting into 1d array and then finding the frequency of each class-------------
        self.pred = self.pred.reshape((self.width*self.height,))
        # storing the frequency of each class present in the predicted mask
        self.pred_count = np.bincount(self.pred, weights=None, minlength=self.classes)  # A

        self.actual = self.actual.reshape((self.width*self.height,))
        # storing the frequency of each class present in the actual mask
        self.actual_count = np.bincount(self.actual, weights=None, minlength=self.classes)  # B
        # -----------------------------------------------------------------------------------------------

        '''there are 21 classes but altogether 21x21=441 possibilities for every pixel
        for example, a pixel may actually belong to class '4' but may be predicted to be in class '3'
        So every pixel will have two features, one of which is actual and the other predicted
        To store both the details, we assign the category to which it belong
        Like in the above mentioned example the pixel belong to category 4-3'''

        # store the category of every pixel
        temp = self.actual * self.classes + self.pred

        # frequency count of temp gives the confusion matrix 'cm' in 1d array format
        self.cm = np.bincount(temp, weights=None, minlength=self.classes**2)
        # reshaping the confusion matrix from 1d array to (no.of classes X no. of classes)
        self.cm = self.cm.reshape((self.classes, self.classes))

    def computeIou(self):
        Nr = np.diag(self.cm)  # A ⋂ B
        #print("Nr:",Nr)
        #print("pred_count:",self.pred_count)
        #print("actual_count:",self.actual_count)
        Dr = self.pred_count + self.actual_count - Nr  # A ⋃ B
        individual_iou = Nr / Dr  # (A ⋂ B)/(A ⋃ B)
        return individual_iou

    def computeMiou(self):
        # the diagonal values of cm correspond to those pixels which belong to same class in both predicted and actual mask
        Nr = np.diag(self.cm)  # A ⋂ B
        Dr = self.pred_count + self.actual_count - Nr  # A ⋃ B
        individual_iou = Nr / Dr  # (A ⋂ B)/(A ⋃ B)
        miou = float(np.nansum(individual_iou)/21)  # nanmean is used to neglect 0/0 case which arise due to absence of any class
        return miou


    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.cm).sum() / self.cm.sum()
        return acc

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.cm) / self.cm.sum(axis=1)
        return classAcc

    # 类别平均像素准确率MPA，对每一类的像素准确率求平均
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc


