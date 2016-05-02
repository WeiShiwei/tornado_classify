"""Univariate features selection."""

# Authors: V. Michel, B. Thirion, G. Varoquaux, A. Gramfort, E. Duchesnay.
#          L. Buitinck, A. Joly
# License: BSD 3 clause

from sklearn.feature_selection import SelectKBest, chi2

select_chi2 = 2000 #1000,'all'
chi2 = SelectKBest(chi2, k=select_chi2)

# """def chi2(X, y):"""
#     """Compute chi-squared statistic for each class/feature combination.

#     This score can be used to select the n_features features with the
#     highest values for the test chi-squared statistic from X, which must
#     contain booleans or frequencies (e.g., term counts in document
#     classification), relative to the classes.

#     Recall that the chi-square test measures dependence between stochastic
#     variables, so using this function "weeds out" the features that are the
#     most likely to be independent of class and therefore irrelevant for
#     classification.

#     Parameters
#     ----------
#     X : {array-like, sparse matrix}, shape = (n_samples, n_features_in)
#         Sample vectors.

#     y : array-like, shape = (n_samples,)
#         Target vector (class labels).

#     Returns
#     -------
#     chi2 : array, shape = (n_features,)
#         chi2 statistics of each feature.
#     pval : array, shape = (n_features,)
#         p-values of each feature.

#     Notes
#     -----
#     Complexity of this algorithm is O(n_classes * n_features).
#     """

# """class SelectKBest(_BaseFilter):"""
#     """Select features according to the k highest scores.

#     Parameters
#     ----------
#     score_func : callable
#         Function taking two arrays X and y, and returning a pair of arrays
#         (scores, pvalues).

#     k : int or "all", optional, default=10
#         Number of top features to select.
#         The "all" option bypasses selection, for use in a parameter search.

#     Attributes
#     ----------
#     scores_ : array-like, shape=(n_features,)
#         Scores of features.

#     pvalues_ : array-like, shape=(n_features,)
#         p-values of feature scores.

#     Notes
#     -----
#     Ties between features with equal scores will be broken in an unspecified
#     way.

#     """