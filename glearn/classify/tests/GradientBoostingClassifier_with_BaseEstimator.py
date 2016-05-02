import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

mnist = datasets.fetch_mldata('MNIST original')
X, y = shuffle(mnist.data, mnist.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.01)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

### works fine when init is None
clf_init = None
print 'Train with clf_init = None'
clf = GradientBoostingClassifier( loss='deviance', learning_rate=0.1,
                             n_estimators=5, subsample=0.3,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             max_depth=3,
                             init=clf_init,
                             random_state=None,
                             max_features=None,
                             verbose=2)
clf.fit(X_train, y_train)
print 'Train with clf_init = None is done :-)'

print 'Train LogisticRegression()'
clf_init = LogisticRegression();
clf_init.fit(X_train, y_train);
print 'Train LogisticRegression() is done'

print 'Train with clf_init = LogisticRegression()'
clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                             n_estimators=5, subsample=0.3,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             max_depth=3,
                             init=clf_init,
                             random_state=None,
                             max_features=None,
                             verbose=2)
clf.fit(X_train, y_train) # <------ ERROR!!!!
print 'Train with clf_init = LogisticRegression() is done'