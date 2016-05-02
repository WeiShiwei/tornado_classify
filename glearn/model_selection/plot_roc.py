"""
=======================================
Receiver Operating Characteristic (ROC)
=======================================

Example of Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality.

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

ROC curves are typically used in binary classification to study the output of
a classifier. In order to extend ROC curve and ROC area to multi-class
or multi-label classification, it is necessary to binarize the output. One ROC
curve can be drawn per label, but one can also draw a ROC curve by considering
each element of the label indicator matrix as a binary prediction
(micro-averaging).

.. note::

    See also :func:`sklearn.metrics.roc_auc_score`,
             :ref:`example_plot_roc_crossval.py`.


Traceback (most recent call last):
  File "linear_svc.py", line 82, in <module>
    main()
  File "linear_svc.py", line 74, in main
    plot_roc(X_train,y_train)
  File "/home/weishiwei/800w_classifier/tornado_classify/glearn/classify/../model_selection/plot_roc.py", line 69, in plot_roc
    y = label_binarize(y, classes=[0, 1, 2])
  File "mtrand.pyx", line 1286, in mtrand.RandomState.randn (numpy/random/mtrand/mtrand.c:9482)
  File "mtrand.pyx", line 1396, in mtrand.RandomState.standard_normal (numpy/random/mtrand/mtrand.c:9781)
  File "mtrand.pyx", line 138, in mtrand.cont0_array (numpy/random/mtrand/mtrand.c:1770)
ValueError: array is too big.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

# import os
# import sys
# sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)) , '..'))
# from datasets import load_files

# vtr=TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
# def plot_roc(data_train):
#     # Import some data to play with
#     # X = roc_data.data
#     import pdb;pdb.set_trace()
#     X = vtr.fit_transform(data_train)
#     y = data_train.target
#     roc(X,y)



def plot_roc(X,y):
    # Import some data to play with
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # import pdb;pdb.set_trace()

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    # import pdb;pdb.set_trace()
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    plot_roc(X,y)