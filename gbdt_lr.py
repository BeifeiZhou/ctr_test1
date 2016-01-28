#This script is the main modeling approach GBDT + LR

import sys
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import label_binarize

print 'Usage: python gbdt_lr.py 30 7'

data = pd.read_csv('../input/gbdt_input.csv', sep = ',', header = None)

X = data[range(data.shape[1]-1)]
y = data[data.shape[1]-1]

#Data sampling for cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)

#n_estimator: tree number
#n_depth: tree depth
n_estimator = int(sys.argv[1])
n_depth = int(sys.argv[2])

#Build model
grd = GradientBoostingClassifier(n_estimators=n_estimator, max_depth=n_depth)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

#Prediction
y_pred_grd_lm = grd_lm.predict_proba(
            grd_enc.transform(grd.apply(X_test)[:, :, 0]))

#Write prediction result
classes = ['impression','click','retargeting','conversion']
predict = pd.DataFrame(y_pred_grd_lm, columns = classes, index = X_test.index)
predict.index.name = 'test index'
predict.insert(0, 'true activity', y_test)
predict.to_csv('../result/predict_gbdt_lr.txt')

#Prepare the prediction result for roc curve analysis
y_test_label = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_label.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_label[:, i], y_pred_grd_lm[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_label.ravel(), y_pred_grd_lm.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot roc curve
with PdfPages('../result/roc.pdf') as pdf:
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for multi-class prediction')
    plt.legend(loc="lower right")
    pdf.savefig() 
