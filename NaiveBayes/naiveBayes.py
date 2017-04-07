'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
        self.useLaplaceSmoothing = useLaplaceSmoothing

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
        self.attributeTables = []
        self.class_counters = []
        self.classes = np.unique(y)
        class_num = len(self.classes)
        self.class_counters = self.getNumForClasses(y)

        '''
        creat a table according to slides
        |----|----|----|
        |attr|clas|prob|
        |----|----|----|
        |    |    |    |
        |    |    |    |
        |    |    |    |
        |----|----|----|
        '''
        for j in range(d):
            this_attribute = X[:, j]
            attribute_values = np.unique(this_attribute)
            attribute_num = len(attribute_values)
            attribute_table = np.zeros([attribute_num*class_num, 3])

            for c in range(class_num):
                attribute_table[c*attribute_num:(c+1)*attribute_num, 0] = attribute_values
                attribute_table[c*attribute_num:(c+1)*attribute_num, 1].fill(self.classes[c])

            for k in range(attribute_num*class_num):
                counter_class = 0
                counter_attribute = 0
                currentValue = attribute_table[k, 0]
                currentLabel = attribute_table[k, 1]

                for i in range(n):
                    if y[i] == currentLabel:
                        counter_class += 1
                for i in range(n):
                    if y[i] == currentLabel and this_attribute[i] == currentValue:
                        counter_attribute += 1

                if self.useLaplaceSmoothing:
                    counter_attribute = counter_attribute + 1
                    counter_class = counter_class + attribute_num
                attribute_table[k, 2] =  counter_attribute / (counter_class*1.0)

            self.attributeTables.append(attribute_table)

    # A helper returns a list containing number of appearance of all classes
    def getNumForClasses(self, y):
        classes = np.unique(y)
        counterList = np.zeros(len(classes))
        for i in range(len(classes)):
            for j in range(len(y)):
                if y[j] == classes[i]:
                    counterList[i] += 1
        return counterList

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape
        result = np.zeros(n)
        for i in range(n):
            argmaxhx = -999            
            predict_class = 0
            #classify the instance            
            for k in self.classes:
                index = 0
                prediction = 0
                temp_class = k
                prediction += np.log(self.class_counters[index]/(1.0*len(self.classes)))
                #choose the attribute with the highest h(x)
                for j in range(d):
                    attribute_table = self.attributeTables[j]
                    #search through the table, find the corresponding probabilty. update h(x)
                    for table_element in attribute_table:
                        if table_element[0] ==X[i,j] and table_element[1] == temp_class:
                            prediction += np.log(table_element[2])
                            # if prediction > argmaxhx:
                            #     argmaxhx = prediction
                            #     predict_class = temp_class ### fix:not working
                if prediction > argmaxhx:
                    argmaxhx = prediction
                    predict_class = temp_class
                index += 1
            result[i] = predict_class

        return result


    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
        n, d = X.shape
        result = np.zeros([n, len(self.classes)])
        #predict a correct label for each instance
        for i in range(n):
            for k in self.classes:
                index = 0
                prob = self.class_counters[index]/(1.0*len(self.classes))
                #choose the correct attribute to use
                for j in range(d):
                    attribute_table = self.attributeTables[j]
                    #select from the table, find the highest probabilty
                    for table_element in attribute_table:
                        if table_element[0] == X[i,j] and table_element[1] == k:
                            prob *= table_element[2]
                result[i,index] = prob
                index += 1
        result = result/sum(result)
        return result
        