#This script is used to calculate the logloss value for the prediction

import pandas as pd
import numpy as np

print 'Usage: python eval.py'

predict = pd.read_csv('../result/predict_gbdt_lr.txt', sep =',')
predict = predict[range(1,6)]
def f(x):
    return x[int(x[0]+1)]

prob = predict.apply(f, axis = 1)
logloss = -1 * np.log(prob).sum()/float(len(prob))
print "The logloss of this prediction is: "+str(logloss)
