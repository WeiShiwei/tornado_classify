tornado-classify
============
利用层级结构进行层次分类（hierarchical text classification or Multi-Label Text Categorization）


Important links
===============
<Introduction to Information Retrieval>信息检索导论

**Note**
1.一种提高分类器在大型层次目录体系下的扩展性的简单方法是进行排他式特征选择（aggressive feature selection）

2.对多个分类器进行投票（voting）、装袋（bagging）及提升（boosting）的研究工作
---可以通过组合这些分类器来略微提升分类精度

3.需要一个自动和手工混合的解决方法来获得足够的分类精确率;
---首先运行一个分类器，然后接受所有高置信度的判定结果，而将那些低置信度的结果放入某个队列供人工浏览来确定。这种过程也会自动地导出新训练文档，这些文档又可以用于机器学习分类器的下一个版本中去

Dependencies
============

scikit-learn is tested to work under Python 2.6, Python 2.7, and Python 3.4.
(using the same codebase thanks to an embedded copy of
`six <http://pythonhosted.org/six/>`_). It should also work with Python 3.3.

The required dependencies to build the software are NumPy >= 1.6.2,
SciPy >= 0.9 and a working C/C++ compiler.

For running the examples Matplotlib >= 1.1.1 is required and for running the
tests you need nose >= 1.1.2.

This configuration matches the Ubuntu Precise 12.04 LTS release from April
2012.

scikit-learn also uses CBLAS, the C interface to the Basic Linear Algebra
Subprograms library. scikit-learn comes with a reference implementation, but
the system CBLAS will be detected by the build system and used if present.
CBLAS exists in many implementations; see `Linear algebra libraries
<http://scikit-learn.org/stable/modules/computational_performance.html#linear-algebra-libraries>`_
for known issues.


jieba
============
pip installl jieba
