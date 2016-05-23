# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:43:20 2016

@author: Administrator
"""

import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

class Classifier():
    """
        train classifier iteratively to get the best one
    """
    def __init__(self):
        self.ratio = 0.38437
        self.threshold = 0.001

    def load_data(self, cpath):
        """
            Load training data, consists of 14 days click data
        """
        print "Loading data: %s" %(time.strftime(ISOTIMEFORMAT, time.localtime()))
        self.click = np.loadtxt(cpath, dtype=np.str, delimiter='|')
        self.click[:, 3] = np.array([time.localtime(float(x))[3] if x.count('.') == 0 else 0 for x in self.click[:, 3]])
        self.click = self.click[:, indices]
        
    def transform(self):
        print "Transforming data: %s" %(time.strftime(ISOTIMEFORMAT, time.localtime()))
        s = set()
        for c in range(len(indices)):
            tmp = [str(c)+'#'+x for x in self.click[:, c]]
            s = s.union(set(tmp))
            
        feat_mapping = dict()
        for index, val in enumerate(s):
            feat_mapping[val] = int(index)
            
        emp = np.empty(np.shape(self.click))
        for c in range(np.shape(self.click)[0]):
            emp[c, :] = np.array([int(feat_mapping[str(k)+'#'+v]) for k, v in enumerate(self.click[c, :])])
        
        self.click = emp
        
    def train_classifier(self):
        ind_ = range(np.shape(self.click)[0])
        max_auc = -np.Inf
        max_index = []
        for epoch in range(20):
            print "Epoch {0}: {1}".format(epoch, time.strftime(ISOTIMEFORMAT, time.localtime()))

            pos_ind = np.random.choice(ind_, int(len(ind_) * self.ratio))
            neg_ind = list(set(range(len(ind_))) - set(pos_ind))
            neg_ind = np.random.choice(neg_ind, len(pos_ind))

            oob_ind = list(set(range(len(ind_))) - set(pos_ind) - set(neg_ind))
            oob_x = self.click

            notrack_pos = self.click[pos_ind, :]
            notrack_neg = self.click[neg_ind, :]
            tr_pos = notrack_pos
            tr_neg = notrack_neg

            old_auc, new_auc = -1.0, 0.0
            lr = LogisticRegression()

            while new_auc - old_auc > self.threshold:
                
                tr_pos_size = np.shape(tr_pos)[0]
                tr_neg_size = np.shape(tr_neg)[0]

                tr_pos_y = np.ones((tr_pos_size, 1))
                tr_neg_y = np.zeros((tr_neg_size, 1))
                
                train = np.vstack((tr_pos, tr_neg))
                train_y = np.ravel(np.vstack((tr_pos_y, tr_neg_y)))
                
                lr.fit(train, train_y)
                
                predicted = lr.predict(train)
                score=roc_auc_score(train_y, predicted)
                new_auc, old_auc = score, new_auc
                print "new auc: {0}, old auc: {1}".format(new_auc, old_auc)
                
                true_pos = [k for k, v in enumerate(train_y) if train_y[k] == predicted[k] and train_y[k] == 1]
                true_neg = [k for k, v in enumerate(train_y) if train_y[k] == predicted[k] and train_y[k] == 0]

                tr_pos = train[true_pos, :]
                tr_neg = train[true_neg, :]

                if new_auc > max_auc:
                    max_index = true_pos
                    max_auc = new_auc
                    break
                
                if len(true_pos) < len(pos_ind):
                    _pos_ind = np.random.choice(oob_ind, len(pos_ind) - len(true_pos))
                    tr_pos = np.vstack((tr_pos, oob_x[_pos_ind, :]))
                    
                    if len(true_neg) < len(pos_ind):
                        _neg_ind = np.random.choice(oob_ind, len(pos_ind) - len(true_neg))
                        tr_neg = np.vstack((tr_neg, oob_x[_neg_ind, :]))

                    oob_ind = list(set(oob_ind) - set(_pos_ind) - set(_neg_ind))
                    if len(oob_ind) == 0:
                        print "no data in out of sample"
                        break
                    
                print "auc: {0}".format(max_auc)
#                print "tr_pos: {0}, tr_neg: {1}".format(np.shape(tr_pos), np.shape(tr_neg))
                
        print "best auc: {0}, index: {1}".format(max_auc, max_index[:50])
        
            
if __name__ == '__main__':
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    indices = [3, 9, 13, 14, 17, 18, 23]
    clf = Classifier()
    clf.load_data("xxxxx")
    clf.transform()
    clf.train_classifier()

