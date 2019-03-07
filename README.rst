GenSVM Python Package
=====================

.. image:: https://travis-ci.org/GjjvdBurg/PyGenSVM.svg?branch=master
    :target: https://travis-ci.org/GjjvdBurg/PyGenSVM

.. image:: https://readthedocs.org/projects/gensvm/badge/?version=latest
   :target: https://gensvm.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


This is the Python package for the GenSVM multiclass classifier by `Gerrit 
J.J. van den Burg <https://gertjanvandenburg.com>`_ and `Patrick J.F. Groenen 
<https://personal.eur.nl/groenen/>`_.

**Important links:**

- Source repository: `https://github.com/GjjvdBurg/PyGenSVM 
  <https://github.com/GjjvdBurg/PyGenSVM>`_.

- Package on PyPI: `https://pypi.org/project/gensvm/ 
  <https://pypi.org/project/gensvm/>`_.

- Journal paper: `GenSVM: A Generalized Multiclass Support Vector Machine 
  <http://www.jmlr.org/papers/v17/14-526.html>`_ JMLR, 17(225):1−42, 2016.

- Package documentation: `Read The Docs 
  <https://gensvm.readthedocs.io/en/latest/>`_.

- There is also an `R package <https://github.com/GjjvdBurg/RGenSVM>`_.

- Or you can directly use `the C library 
  <https://github.com/GjjvdBurg/GenSVM>`_.


Installation
------------

**Before** GenSVM can be installed, a working NumPy installation is required.  
Please see `the installation instructions for NumPy 
<https://docs.scipy.org/doc/numpy-1.13.0/user/install.html>`_, then install 
GenSVM using the instructions below.

GenSVM can be easily installed through pip:

.. code:: bash

    pip install gensvm

If you encounter any errors, please open an issue on `GitHub 
<https://github.com/GjjvdBurg/PyGenSVM>`_.

Citing
------

If you use this package in your research please cite the paper, for instance 
using the following BibTeX entry::

    @article{JMLR:v17:14-526,
      author  = {{van den Burg}, G. J. J. and Groenen, P. J. F.},
      title   = {{GenSVM}: A Generalized Multiclass Support Vector Machine},
      journal = {Journal of Machine Learning Research},
      year    = {2016},
      volume  = {17},
      number  = {225},
      pages   = {1-42},
      url     = {http://jmlr.org/papers/v17/14-526.html}
    }

Usage
-----

The package contains two classes to fit the GenSVM model: `GenSVM`_ and 
`GenSVMGridSearchCV`_.  These classes respectively fit a single GenSVM model 
or fit a series of models for a parameter grid search. The interface to these 
classes is the same as that of classifiers in `Scikit-Learn`_  so users 
familiar with Scikit-Learn should have no trouble using this package.  Below 
we will show some examples of using the GenSVM classifier and the 
GenSVMGridSearchCV class in practice.

In the examples we assume that we have loaded the `iris dataset
<http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>`_ 
from Scikit-Learn as follows:

.. code:: python

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import MaxAbsScaler
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> scaler = MaxAbsScaler().fit(X_train)
    >>> X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

Note that we scale the data using the `MaxAbsScaler 
<http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html>`_ 
function. This scales the columns of the data matrix to ``[-1, 1]`` without 
breaking sparsity. Scaling the dataset can have a significant effect on the 
computation time of GenSVM and is `generally recommended for SVMs 
<https://stats.stackexchange.com/q/65094>`_.


Example 1: Fitting a single GenSVM model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's start by fitting the most basic GenSVM model on the training data:

.. code:: python

    >>> from gensvm import GenSVM
    >>> clf = GenSVM()
    >>> clf.fit(X_train, y_train)
    GenSVM(coef=0.0, degree=2.0, epsilon=1e-06, gamma='auto', kappa=0.0,
    kernel='linear', kernel_eigen_cutoff=1e-08, lmd=1e-05,
    max_iter=100000000.0, p=1.0, random_state=None, verbose=0,
    weights='unit')


With the model fitted, we can predict the test dataset:

.. code:: python

    >>> y_pred = clf.predict(X_test)

Next, we can compute a score for the predictions. The GenSVM class has a 
``score`` method which computes the `accuracy_score 
<http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`_ 
for the predictions. In the GenSVM paper, the `adjusted Rand index 
<https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`_ is often used 
to compare performance. We illustrate both options below (your results may be 
different depending on the exact train/test split):

.. code:: python

    >>> clf.score(X_test, y_test)
    1.0
    >>> from sklearn.metrics import adjusted_rand_score
    >>> adjusted_rand_score(clf.predict(X_test), y_test)
    1.0

We can try this again by changing the model parameters, for instance we can 
turn on verbosity and use the Euclidean norm in the GenSVM model by setting ``p = 2``:

.. code:: python

    >>> clf2 = GenSVM(verbose=True, p=2)
    >>> clf2.fit(X_train, y_train)
    Starting main loop.
    Dataset:
        n = 112
        m = 4
        K = 3
    Parameters:
        kappa = 0.000000
        p = 2.000000
        lambda = 0.0000100000000000
        epsilon = 1e-06
    
    iter = 0, L = 3.4499531579689533, Lbar = 7.3369415851139745, reldiff = 1.1266786095824437
    ...
    Optimization finished, iter = 4046, loss = 0.0230726364692517, rel. diff. = 0.0000009998645783
    Number of support vectors: 9
    GenSVM(coef=0.0, degree=2.0, epsilon=1e-06, gamma='auto', kappa=0.0,
        kernel='linear', kernel_eigen_cutoff=1e-08, lmd=1e-05,
        max_iter=100000000.0, p=2, random_state=None, verbose=True,
        weights='unit')

For other parameters that can be tuned in the GenSVM model, see `GenSVM`_.


Example 2: Fitting a GenSVM model with a "warm start"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One of the key features of the GenSVM classifier is that training can be 
accelerated by using so-called "warm-starts". This way the optimization can be 
started in a location that is closer to the final solution than a random 
starting position would be. To support this, the ``fit`` method of the GenSVM 
class has an optional ``seed_V`` parameter. We'll illustrate how this can be 
used below.

We start with relatively large value for the ``epsilon`` parameter in the 
model. This is the stopping parameter that determines how long the 
optimization continues (and therefore how exact the fit is).

.. code:: python

    >>> clf1 = GenSVM(epsilon=1e-3)
    >>> clf1.fit(X_train, y_train)
    ...
    >>> clf1.n_iter_
    163

The ``n_iter_`` attribute tells us how many iterations the model did. Now, we 
can use the solution of this model to start the training for the next model:

.. code:: python

    >>> clf2 = GenSVM(epsilon=1e-8)
    >>> clf2.fit(X_train, y_train, seed_V=clf1.combined_coef_)
    ...
    >>> clf2.n_iter_
    3196

Compare this to a model with the same stopping parameter, but without the warm 
start:

.. code:: python

    >>> clf2.fit(X_train, y_train)
    ...
    >>> clf2.n_iter_
    3699

So we saved about 500 iterations! This effect will be especially significant 
with large datasets and when you try out many parameter configurations.  
Therefore this technique is built into the `GenSVMGridSearchCV`_ class that 
can be used to do a grid search of parameters.


Example 3: Running a GenSVM grid search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often when we're fitting a machine learning model such as GenSVM, we have to 
try several parameter configurations to figure out which one performs best on 
our given dataset. This is usually combined with `cross validation 
<http://scikit-learn.org/stable/modules/cross_validation.html>`_ to avoid 
overfitting. To do this efficiently and to make use of warm starts, the 
`GenSVMGridSearchCV`_ class is available. This class works in the same way as 
the `GridSearchCV 
<http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_ 
class of `Scikit-Learn`_, but uses the GenSVM C library for speed.

To do a grid search, we first have to define the parameters that we want to 
vary and what values we want to try:

.. code:: python

    >>> from gensvm import GenSVMGridSearchCV
    >>> param_grid = {'p': [1.0, 2.0], 'lmd': [1e-8, 1e-6, 1e-4, 1e-2, 1.0], 'kappa': [-0.9, 0.0] }

For the values that are not varied in the parameter grid, the default values 
will be used. This means that if you want to change a specific value (such as 
``epsilon`` for instance), you can add this to the parameter grid as a 
parameter with a single value to try (e.g. ``'epsilon': [1e-8]``).

Running the grid search is now straightforward:

.. code:: python

    >>> gg = GenSVMGridSearchCV(param_grid)
    >>> gg.fit(X_train, y_train)
    GenSVMGridSearchCV(cv=None, iid=True,
          param_grid={'p': [1.0, 2.0], 'lmd': [1e-06, 0.0001, 0.01, 1.0], 'kappa': [-0.9, 0.0]},
          refit=True, return_train_score=True, scoring=None, verbose=0)

Note that if we have set ``refit=True`` (the default), then we can use the 
`GenSVMGridSearchCV`_ instance to predict or score using the best estimator 
found in the grid search:

.. code:: python

    >>> y_pred = gg.predict(X_test)
    >>> gg.score(X_test, y_test)
    1.0

A nice feature borrowed from `Scikit-Learn`_ is that the results from the grid 
search can be represented as a ``pandas`` DataFrame:

.. code:: python

    >>> from pandas import DataFrame
    >>> df = DataFrame(gg.cv_results_)

This can make it easier to explore the results of the grid search.

Known Limitations
-----------------

The following are known limitations that are on the roadmap for a future 
release of the package. If you need any of these features, please vote on them 
on the linked GitHub issues (this can make us add them sooner!).

1. `Support for sparse matrices 
   <https://github.com/GjjvdBurg/PyGenSVM/issues/1>`_. NumPy supports sparse 
   matrices, as does the GenSVM C library. Getting them to work together 
   requires some time. In the meantime, if you really want to use sparse data 
   with GenSVM (this can lead to significant speedups!), check out the GenSVM 
   C library.
2. `Specification of class misclassification weights 
   <https://github.com/GjjvdBurg/PyGenSVM/issues/3>`_. Currently, incorrectly 
   classification an object from class A to class C is as bad as incorrectly 
   classifying an object from class B to class C. Depending on the 
   application, this may not be the desired effect. Adding class 
   misclassification weights can solve this issue.

Questions and Issues
--------------------

If you have any questions or encounter any issues with using this package, 
please ask them on `GitHub <https://github.com/GjjvdBurg/PyGenSVM>`_.

License
-------

This package is licensed under the GNU General Public License version 3.  

Copyright G.J.J. van den Burg, excluding the sections of the code that are 
explicitly marked to come from Scikit-Learn.

.. _Scikit-Learn:
    http://scikit-learn.org/stable/index.html

.. _GenSVM:
    https://gensvm.readthedocs.io/en/latest/#gensvm

.. _GenSVMGridSearchCV:
    https://gensvm.readthedocs.io/en/latest/#gensvmgridsearchcv
