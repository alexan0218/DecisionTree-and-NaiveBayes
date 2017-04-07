'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
#from sklearn import normalize

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO

        #Use the algorithm 2 in http://web.stanford.edu/~hastie/Papers/samme.pdf
        n,d = X.shape
        self.classes = np.unique(y)
        self.classes_num = len(self.classes)
        self.beta = np.zeros(self.numBoostingIters)
        self.classification = []        
        self.weights = np.zeros((n))

        #step 1: initialize weights
        for i in range(n):
            self.weights[i] = 1.0/n
        #step 2: 
        for i in range(self.numBoostingIters):

            #step a
            clf = tree.DecisionTreeClassifier()
            clf.max_depth = self.maxTreeDepth
            clf = clf.fit(X,y,sample_weight = self.weights)
            #store the nth classifier 
            predictY = clf.predict(X)
            self.classification.append(clf)

            #step b
            numerator = 0 
            for j in range(n):
                if (y[j] != predictY[j]):
                    numerator += self.weights[j]
                    err = numerator / sum(self.weights)

            #step c
            log =  (1-err)/err
            beta = np.log(log) + np.log(len(self.classes-1))
            self.beta[i] = beta

            #step d
            for j in range(n):
                if y[j] != predictY[j]:
                    self.weights[j] *= np.exp(beta)      

            #step e
            self.weights = self.weights/sum(self.weights)
            # print self.weights
            #self.weights = normalize(self.weights, axis = 1, norm = 'l1')

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
        n,d = X.shape
        pred_result = np.zeros(n)
        #predict every single instance
        for i in range(n):            
            argmaxBeta = 0
            this_Class = 0

            print "finished: " + str(i) + " : " + str(n)

            #choose class label
            for j in range(self.classes_num):
                sigma = 0
                for k in range(self.numBoostingIters):
                    #use pre-stored classifications to predict
                    if self.classification[k].predict(X)[i] == self.classes[j]:
                        sigma += self.beta[k]
                        #update the prediction
                        if sigma > argmaxBeta:
                            argmaxBeta = sigma
                            this_Class = self.classes[j]
                            pred_result[i] = this_Class
            
        return pred_result