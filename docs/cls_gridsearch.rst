GenSVMGridSearchCV
==================

.. py:class:: GenSVMGridSearchCV(param_grid='tiny', scoring=None, iid=True, cv=None, refit=True, verbose=0, return_train_score=True)
   :noindex:
   :module: gensvm.gridsearch

   GenSVM cross validated grid search
   
   This class implements efficient GenSVM grid search with cross validation.
   One of the strong features of GenSVM is that seeding the classifier
   properly can greatly reduce total training time. This class ensures that
   the grid search is done in the most efficient way possible.
   
   The implementation of this class is based on the `GridSearchCV`_ class in
   scikit-learn. The documentation of the various parameters is therefore
   mostly the same. This is done to provide the user with a familiar and
   easy-to-use interface to doing a grid search with GenSVM. A separate class
   was needed to benefit from the fast low-level C implementation of grid
   search in the GenSVM library.
   
   :param param_grid: If a string, it must be either 'tiny', 'small', or 'full' to load the
                      predefined parameter grids (see the functions :func:`load_grid_tiny`,
                      :func:`load_grid_small`, and :func:`load_grid_full`).
   
                      Otherwise, a dictionary of parameter names (strings) as keys and lists
                      of parameter settings to evaluate as values, or a list of such dicts.
                      The GenSVM model will be evaluated at all combinations of the
                      parameters.
   :type param_grid: string, dict, or list of dicts
   :param scoring: A single string (see :ref:`scoring_parameter`) or a callable (see
                   :ref:`scoring`) to evaluate the predictions on the test set.
   
                   For evaluating multiple metrics, either give a list of (unique) strings
                   or a dict with names as keys and callables as values.
   
                   NOTE that when using custom scorers, each scorer should return a single
                   value. Metric functions returning a list/array of values can be wrapped
                   into multiple scorers that return one value each.
   
                   If None, the `accuracy_score`_ is used.
   :type scoring: string, callable, list/tuple, dict or None
   :param iid: If True, the data is assumed to be identically distributed across the
               folds, and the loss minimized is the total loss per sample and not the
               mean loss across the folds.
   :type iid: boolean, default=True
   :param cv: Determines the cross-validation splitting strategy. Possible inputs for
              cv are:
   
                - None, to use the default 5-fold cross validation,
                - integer, to specify the number of folds in a `(Stratified)KFold`,
                - An object to be used as a cross-validation generator.
                - An iterable yielding train, test splits.
   
              For integer/None inputs, :class:`StratifiedKFold
              <sklearn.model_selection.StratifiedKFold>` is used.  In all other
              cases, :class:`KFold <sklearn.model_selection.KFold>` is used.
   
              Refer to the `scikit-learn User Guide on cross validation`_ for the
              various strategies that can be used here.
   
              NOTE: At the moment, the ShuffleSplit and StratifiedShuffleSplit are
              not supported in this class. If you need these, you can use the GenSVM
              classifier directly with the GridSearchCV object from scikit-learn.
              (these methods require significant changes in the low-level library
              before they can be supported).
   :type cv: int, cross-validation generator or an iterable, optional
   :param refit: Refit the GenSVM estimator with the best found parameters on the whole
                 dataset.
   
                 For multiple metric evaluation, this needs to be a string denoting the
                 scorer to be used to find the best parameters for refitting the
                 estimator at the end.
   
                 The refitted estimator is made available at the `:attr:best_estimator_
                 <.GenSVMGridSearchCV.best_estimator_>` attribute and allows the user to
                 use the :func:`~GenSVMGridSearchCV.predict` method directly on this
                 :class:`.GenSVMGridSearchCV` instance.
   
                 Also for multiple metric evaluation, the attributes :attr:`best_index_
                 <.GenSVMGridSearchCV.best_index_>`, :attr:`best_score_
                 <.GenSVMGridSearchCV.best_score_>` and :attr:`best_params_
                 <.GenSVMGridSearchCV:best_params_>` will only be available if ``refit``
                 is set and all of them will be determined w.r.t this specific scorer.
   
                 See ``scoring`` parameter to know more about multiple metric
                 evaluation.
   :type refit: boolean, or string, default=True
   :param verbose: Controls the verbosity: the higher, the more messages.
   :type verbose: integer
   :param return_train_score: If ``False``, the :attr:`cv_results_ <.GenSVMGridSearchCV.cv_results_>`
                              attribute will not include training scores.
   :type return_train_score: boolean, default=True
   
   .. rubric:: Examples
   
   >>> from gensvm import GenSVMGridSearchCV
   >>> from sklearn.datasets import load_iris
   >>> iris = load_iris()
   >>> param_grid = {'p': [1.0, 2.0], 'kappa': [-0.9, 0.0, 1.0]}
   >>> clf = GenSVMGridSearchCV(param_grid)
   >>> clf.fit(iris.data, iris.target)
   GenSVMGridSearchCV(cv=None, iid=True,
         param_grid={'p': [1.0, 2.0], 'kappa': [-0.9, 0.0, 1.0]},
         refit=True, return_train_score=True, scoring=None, verbose=0)
   
   .. attribute:: cv_results_
   
      *dict of numpy (masked) ndarrays* -- A dict with keys as column headers and values as columns, that can be
      imported into a pandas `DataFrame`_.
   
      For instance the below given table
   
      +------------+-----------+------------+-----------------+---+---------+
      |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
      +============+===========+============+=================+===+=========+
      |  'poly'    |     --    |      2     |        0.8      |...|    2    |
      +------------+-----------+------------+-----------------+---+---------+
      |  'poly'    |     --    |      3     |        0.7      |...|    4    |
      +------------+-----------+------------+-----------------+---+---------+
      |  'rbf'     |     0.1   |     --     |        0.8      |...|    3    |
      +------------+-----------+------------+-----------------+---+---------+
      |  'rbf'     |     0.2   |     --     |        0.9      |...|    1    |
      +------------+-----------+------------+-----------------+---+---------+
   
      will be represented by a ``cv_results_`` dict of::
   
          {
          'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                       mask = [False False False False]...)
          'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                      mask = [ True  True False False]...),
          'param_degree': masked_array(data = [2.0 3.0 -- --],
                                       mask = [False False  True  True]...),
          'split0_test_score'  : [0.8, 0.7, 0.8, 0.9],
          'split1_test_score'  : [0.82, 0.5, 0.7, 0.78],
          'mean_test_score'    : [0.81, 0.60, 0.75, 0.82],
          'std_test_score'     : [0.02, 0.01, 0.03, 0.03],
          'rank_test_score'    : [2, 4, 3, 1],
          'split0_train_score' : [0.8, 0.9, 0.7],
          'split1_train_score' : [0.82, 0.5, 0.7],
          'mean_train_score'   : [0.81, 0.7, 0.7],
          'std_train_score'    : [0.03, 0.03, 0.04],
          'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
          'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
          'mean_score_time'    : [0.007, 0.06, 0.04, 0.04],
          'std_score_time'     : [0.001, 0.002, 0.003, 0.005],
          'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
          }
   
      NOTE:
   
      The key ``'params'`` is used to store a list of parameter settings
      dicts for all the parameter candidates.
   
      The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
      ``std_score_time`` are all in seconds.
   
      For multi-metric evaluation, the scores for all the scorers are
      available in the :attr:`cv_results_ <.GenSVMGridSearchCV.cv_results_>`
      dict at the keys ending with that scorer's name (``'_<scorer_name>'``)
      instead of ``'_score'`` shown above. ('split0_test_precision',
      'mean_train_precision' etc.)
   
   .. attribute:: best_estimator_
   
      *estimator or dict* -- Estimator that was chosen by the search, i.e. estimator which gave
      highest score (or smallest loss if specified) on the left out data. Not
      available if ``refit=False``.
   
      See ``refit`` parameter for more information on allowed values.
   
   .. attribute:: best_score_
   
      *float* -- Mean cross-validated score of the best_estimator
   
      For multi-metric evaluation, this is present only if ``refit`` is
      specified.
   
   .. attribute:: best_params_
   
      *dict* -- Parameter setting that gave the best results on the hold out data.
   
      For multi-metric evaluation, this is present only if ``refit`` is
      specified.
   
   .. attribute:: best_index_
   
      *int* -- The index (of the ``cv_results_`` arrays) which corresponds to the best
      candidate parameter setting.
   
      The dict at ``search.cv_results_['params'][search.best_index_]`` gives
      the parameter setting for the best model, that gives the highest mean
      score (``search.best_score_``).
   
      For multi-metric evaluation, this is present only if ``refit`` is
      specified.
   
   .. attribute:: scorer_
   
      *function or a dict* -- Scorer function used on the held out data to choose the best parameters
      for the model.
   
      For multi-metric evaluation, this attribute holds the validated
      ``scoring`` dict which maps the scorer key to the scorer callable.
   
   .. attribute:: n_splits_
   
      *int* -- The number of cross-validation splits (folds/iterations).
   
   .. rubric:: Notes
   
   The parameters selected are those that maximize the score of the left out
   data, unless an explicit score is passed in which case it is used instead.
   
   .. seealso::
   
      `ParameterGrid`_:
          Generates all the combinations of a hyperparameter grid.
   
      :class:`.GenSVM`:
          The GenSVM classifier
   
      .. _GridSearchCV:
          http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
      .. _accuracy_score:
          http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
      .. _scikit-learn User Guide on cross validation:
          http://scikit-learn.org/stable/modules/cross_validation.html
   
      .. _ParameterGrid:
          http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterGrid.html
      .. _DataFrame:
          https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
   
   
   .. py:method:: GenSVMGridSearchCV.fit(X, y, groups=None)
      :noindex:
      :module: gensvm.gridsearch
   
      Run GenSVM grid search with all sets of parameters
      
      :param X: Training data, where n_samples is the number of observations and
                n_features is the number of features.
      :type X: array-like, shape = (n_samples, n_features)
      :param y: Target vector for the training data.
      :type y: array-like, shape = (n_samples, )
      :param groups: Group labels for the samples used while splitting the dataset into
                     train/test sets.
      :type groups: array-like, with shape (n_samples, ), optional
      
      :returns: **self** -- Return self.
      :rtype: object
      
   
   .. py:method:: GenSVMGridSearchCV.predict(X, trainX=None)
      :noindex:
      :module: gensvm.gridsearch
   
      Predict the class labels on the test data
      
      :param X: Test data, where n_samples is the number of observations and
                n_features is the number of features.
      :type X: array-like, shape = (n_samples, n_features)
      :param trainX: Only for nonlinear prediction with kernels: the training data used
                     to train the model.
      :type trainX: array, shape = [n_train_samples, n_features]
      
      :returns: **y_pred** -- Predicted class labels of the data in X.
      :rtype: array-like, shape = (n_samples, )
      
   
   .. py:method:: GenSVMGridSearchCV.score(X, y)
      :noindex:
      :module: gensvm.gridsearch
   
      Compute the score on the test data given the true labels
      
      :param X: Test data, where n_samples is the number of observations and
                n_features is the number of features.
      :type X: array-like, shape = (n_samples, n_features)
      :param y: True labels for the test data.
      :type y: array-like, shape = (n_samples, )
      
      :returns: **score**
      :rtype: float
      
