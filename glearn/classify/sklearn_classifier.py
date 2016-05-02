#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
#import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
# from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# ----------------------------------------------------------------------------------
from sklearn import naive_bayes
from sklearn import svm

class MultinomialNB(naive_bayes.MultinomialNB):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    fit_prior : boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size (n_classes,)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.

    intercept_ : property
        Mirrors ``class_log_prior_`` for interpreting MultinomialNB
        as a linear model.

    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    coef_ : property
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNB
        as a linear model.

    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, y)
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2]))
    [3]

    Notes
    -----
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/
        naive-bayes-text-classification-1.html
    """
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super(MultinomialNB, self).__init__(alpha=1.0, fit_prior=True, class_prior=None)

class LinearSVC(svm.LinearSVC):
    """Linear Support Vector Classification.

    Similar to SVC with parameter kernel='linear', but implemented in terms of
    liblinear rather than libsvm, so it has more flexibility in the choice of
    penalties and loss functions and should scale better (to large numbers of
    samples).

    This class supports both dense and sparse input and the multiclass support
    is handled according to a one-vs-the-rest scheme.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    loss : string, 'l1' or 'l2' (default='l2')
        Specifies the loss function. 'l1' is the hinge loss (standard SVM)
        while 'l2' is the squared hinge loss.

    penalty : string, 'l1' or 'l2' (default='l2')
        Specifies the norm used in the penalization. The 'l2'
        penalty is the standard used in SVC. The 'l1' leads to `coef_`
        vectors that are sparse.

    dual : bool, (default=True)
        Select the algorithm to either solve the dual or primal
        optimization problem. Prefer dual=False when n_samples > n_features.

    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria

    multi_class: string, 'ovr' or 'crammer_singer' (default='ovr')
        Determines the multi-class strategy if `y` contains more than
        two classes.
        `ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`
        optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from an theoretical perspective
        as it is consistent it is seldom used in practice and rarely leads to
        better accuracy and is more expensive to compute.
        If `crammer_singer` is chosen, the options loss, penalty and dual will
        be ignored.

    fit_intercept : boolean, optional (default=True)
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    intercept_scaling : float, optional (default=1)
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.

    verbose : int, default: 0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    max_iter : int, default 1000
        The maximum number of iterations to be run.

    Attributes
    ----------
    coef_ : array, shape = [n_features] if n_classes == 2 \
            else [n_classes, n_features]
        Weights asigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    The underlying implementation (liblinear) uses a sparse internal
    representation for the data that will incur a memory copy.

    **References:**
    `LIBLINEAR: A Library for Large Linear Classification
    <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__

    See also
    --------
    SVC
        Implementation of Support Vector Machine classifier using libsvm:
        the kernel can be non-linear but its SMO algorithm does not
        scale to large number of samples as LinearSVC does.

        Furthermore SVC multi-class mode is implemented using one
        vs one scheme while LinearSVC uses one vs the rest. It is
        possible to implement one vs the rest with SVC by using the
        :class:`sklearn.multiclass.OneVsRestClassifier` wrapper.

        Finally SVC can fit dense data without memory copy if the input
        is C-contiguous. Sparse data will still incur memory copy though.

    sklearn.linear_model.SGDClassifier
        SGDClassifier can optimize the same cost function as LinearSVC
        by adjusting the penalty and loss parameters. In addition it requires
        less memory, allows incremental (online) learning, and implements
        various loss functions and regularization regimes.

    """
    def __init__(self, penalty='l2', loss='l2', dual=True, tol=1e-4, C=1.0,
                 multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                 class_weight=None, verbose=0, random_state=None, max_iter=1000):
        super(LinearSVC, self).__init__(
            penalty=penalty, loss=loss, dual=dual, tol=tol, C=C,
            multi_class=multi_class, fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight, verbose=verbose,
            random_state=random_state) #TypeError: __init__() got an unexpected keyword argument 'max_iter'

class SVC(svm.SVC):
    """C-Support Vector Classification.

    The implementations is a based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to dataset with more than a couple of 10000 samples.

    The multiclass support is handled according to a one-vs-one scheme.

    For details on the precise mathematical formulation of the provided
    kernel functions and how `gamma`, `coef0` and `degree` affect each,
    see the corresponding section in the narrative documentation:
    :ref:`svm_kernels`.

    .. The narrative documentation is available at http://scikit-learn.org/

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
         a callable.
         If none is given, 'rbf' will be used. If a callable is given it is
         used to precompute the kernel matrix.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 0.0 then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    probability: boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.

    shrinking: boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB)

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one. The 'auto' mode uses the values of y to
        automatically adjust weights inversely proportional to
        class frequencies.

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data for probability estimation.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Index of support vectors.

    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.

    n_support_ : array-like, dtype=int32, shape = [n_class]
        number of support vector for each class.

    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function. \
        For multiclass, coefficient for all 1-vs-1 classifiers. \
        The layout of the coefficients in the multiclass case is somewhat \
        non-trivial. See the section about multi-class classification in the \
        SVM section of the User Guide for details.

    coef_ : array, shape = [n_class-1, n_features]
        Weights asigned to the features (coefficients in the primal
        problem). This is only available in the case of linear kernel.

        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`

    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = SVC()
    >>> clf.fit(X, y) #doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
        gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    See also
    --------
    SVR
        Support Vector Machine for Regression implemented using libsvm.

    LinearSVC
        Scalable Linear Support Vector Machine for classification
        implemented using liblinear. Check the See also section of
        LinearSVC for more comparison element.

    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, random_state=None):
        # super(SVC, self).__init__(
        #     'c_svc', kernel, degree, gamma, coef0, tol, C, 0., 0., shrinking,
        #     probability, cache_size, class_weight, verbose, max_iter,
        #     random_state)
        super(SVC, self).__init__(C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, random_state=None)





# ------------------------------------------------------------------------------------------------
class L1LinearSVC(svm.LinearSVC):
    """for example:contrast to LinearSVC"""
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1",
                                      dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)
